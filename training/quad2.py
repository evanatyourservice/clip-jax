"""
quad_pipeline3.py — pipeline-sharded PSGD-QUAD (JAX/Optax)

This optimizer implements PSGD-QUAD with an execution model designed for
pipeline parallelism. The core idea is to (1) reshape each gradient into a
matrix well-suited for dense/diagonal preconditioning, (2) block-partition it
to create a *single* leading batch dimension, and (3) vmap the QUAD update
over that batch dimension. For throughput, *all dense blocks from all layers*
are concatenated once per step and sharded along a named mesh axis.

High-level flow per step
------------------------
1) **Scanning & grouping**:
   - If a layer is "scanned", we keep its leading scan batch; otherwise we add
     a dummy batch dim so every leaf has a leading batch.
   - Leaves are grouped by shape (ignoring any batch dimension):
     • DENSE: All dimensions ≤ `max_size_dense` (and not 1D-like). These get
       a dense-dense preconditioner after being reshaped to 2D.
     • LARGE: At least one dimension > `max_size_dense`. Large dimensions get
       diagonal preconditioning; other dims are merged for dense preconditioning.
     • ONE_D: 1D/1D-like tensors (e.g., biases, scales), which get a single
       diagonal preconditioner.

2) **Prepare (always on)**:
   - Merge non-batch dims to the most square 1D/2D shape.
   - Build a BlockPartitioner that splits only non-diagonal axes by `block_size`.
   - Pad only dense axes to their next multiple of `block_size`.
   - Stack partitions → shape `[batch*stack, block..., block...]`.

3) **Sharding**:
   - DENSE: concatenate all leaves along the leading dim; pad to a multiple of
     `pipeline_axis_size`; set sharding constraint `P(pipeline_axis_name)`.
     We keep Q/L for the concatenated dense path *in this sharded form*
     between steps.
   - LARGE: pad each leaf's leading dim to `pipeline_axis_size` and shard it.
   - ONE_D: replicated (no sharding needed).

4) **QUAD update (vmapped)**:
   - For each sample in the leading batch, compute Lipschitz (in f32), update
     dense/diag preconditioners Q, and produce preconditioned grads Pg.
   - Global RMS clamp to 1.1; optional Adam-style scale (/5) for compatibility.

5) **Unprepare**:
   - Split `[batch*stack,...]` back to `[batch, stack,...]`, unpad/unblock,
     reshape to original param shapes, and drop dummy scan dims.

Other design choices
--------------------
• Single `dtype` (bf16 or f32) is used for momentum and Q; L is always f32.
• Momentum buffers keep *original* param shapes/specs.
• Scalars `()` are handled as `(1,)` internally and restored on exit.
• Flax `nn.Partitioned` is unboxed/boxed around the update.
• Final preconditioned updates are re-constrained to `params_partition_specs`
  if provided.

Author: https://github.com/evanatyourservice
"""

from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union
import numpy as np

import jax
from jax import numpy as jnp, vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

try:
    import flax.linen as nn

    HAVE_FLAX = True
except Exception:
    HAVE_FLAX = False
    nn = None  # type: ignore


PartitionSpecTree = TypeVar(
    "PartitionSpecTree", bound=Union[PartitionSpec, List[PartitionSpec], Tuple[PartitionSpec, ...], dict, list, tuple]
)


def _shard(x, spec: Optional[Union[PartitionSpec, PartitionSpecTree]]):
    if spec is None:
        return x
    try:
        leaves, tdef = jax.tree.flatten(x)
    except Exception:
        return with_sharding_constraint(x, spec if isinstance(spec, PartitionSpec) else spec)
    if isinstance(spec, PartitionSpec):
        return tdef.unflatten([with_sharding_constraint(v, spec) for v in leaves])
    try:
        specs, sdef = jax.tree.flatten(spec)
        if sdef == tdef:
            out = [with_sharding_constraint(v, s) if s is not None else v for v, s in zip(leaves, specs)]
            return tdef.unflatten(out)
    except Exception:
        pass
    return tdef.unflatten([with_sharding_constraint(v, spec) for v in leaves])  # fallback


def _merge_dims_and_diag(shape: Tuple[int, ...], max_dense: int) -> Tuple[List[int], List[bool]]:
    """
    merge dims per rules:
      • 1d-like -> single diag axis
      • no large dims (> max_dense) -> merge to two most-square dense axes
      • otherwise keep every large dim as a diag axis and merge only consecutive
        dense runs around/between them into single dense axes (order preserved)

    returns (merged_shape, diag_flags_per_axis)
    """
    if len(shape) == 0:
        return [1], [True]
    if len(shape) == 1:
        return [int(shape[0])], [True]
    dims = list(map(int, shape))
    # 1d-like
    non1 = [d for d in dims if d != 1]
    if len(non1) <= 1:
        return [int(np.prod(dims))], [True]
    large_mask = [d > max_dense for d in dims]
    has_large = any(large_mask)

    if not has_large:
        # all dense, merge to most square
        if len(dims) == 2:
            return [dims[0], dims[1]], [False, False]
        best_ratio, best_split = float("inf"), 1
        for s in range(1, len(dims)):
            lp, rp = int(np.prod(dims[:s])), int(np.prod(dims[s:]))
            r = max(lp, rp) / min(lp, rp)
            if r < best_ratio:
                best_ratio, best_split = r, s
        return [int(np.prod(dims[:best_split])), int(np.prod(dims[best_split:]))], [False, False]

    # has large dim, keep each large dim, merge only consecutive dense dims
    merged: List[int] = []
    diag_flags: List[bool] = []
    run_prod = 1
    in_run = False
    for d, is_large in zip(dims, large_mask):
        if is_large:
            if in_run:
                merged.append(int(run_prod))
                diag_flags.append(False)
                run_prod, in_run = 1, False
            merged.append(int(d))
            diag_flags.append(True)
        else:
            run_prod *= d
            in_run = True
    if in_run:
        merged.append(int(run_prod))
        diag_flags.append(False)

    return merged, diag_flags


def _is_1d_like(shape: Tuple[int, ...]) -> bool:
    if len(shape) <= 1:
        return True
    return int(np.prod(shape)) == int(np.max(shape))


def _pad_to_multiple(x: Optional[jax.Array], k: int) -> Optional[jax.Array]:
    if x is None or k <= 1:
        return x
    pad = (-x.shape[0]) % k
    return jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1)) if pad else x


def _pad_shard_leading(x: Optional[jax.Array], axis_name: Optional[str], axis_size: int) -> Optional[jax.Array]:
    if x is None or axis_name is None:
        return x
    return _shard(_pad_to_multiple(x, axis_size), PartitionSpec(axis_name))


def _pad_leading_with_identity(arr: Optional[jax.Array], k: int) -> Optional[jax.Array]:
    if arr is None or k <= 1:
        return arr
    pad = (-arr.shape[0]) % k
    if pad == 0:
        return arr
    if arr.ndim == 3:
        n = arr.shape[1]
        tail = jnp.broadcast_to(jnp.eye(n, dtype=arr.dtype), (pad, n, n))
    elif arr.ndim == 2:
        tail = jnp.ones((pad, arr.shape[1]), dtype=arr.dtype)
    else:
        tail = jnp.ones((pad,) + arr.shape[1:], dtype=arr.dtype)
    return jnp.concatenate([arr, tail], axis=0)


def _pad_shard_Q_axes_list(
    lst: Optional[List[jax.Array]], axis_name: Optional[str], axis_size: int
) -> Optional[List[jax.Array]]:
    if lst is None:
        return None
    out: List[jax.Array] = []
    for x in lst:
        x = _pad_leading_with_identity(x, axis_size)
        x = _shard(x, PartitionSpec(axis_name)) if axis_name is not None else x
        out.append(x)
    return out


class BlockPartitioner:
    def __init__(self, param_shape: Tuple[int, ...], block_size: int, dim_diag: List[bool]):
        param_shape = tuple(map(int, param_shape))
        assert len(dim_diag) == len(param_shape)
        self._shape = param_shape
        self._splits = []
        split_sizes = []
        for i, (d, diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not diag:
                nsplit = (d - 1) // block_size
                idx = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.full(nsplit + 1, block_size, np.int32)
                sizes[-1] = d - idx[-1]
                self._splits.append((i, idx))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes
        single = [int(s[0]) for s in split_sizes]
        padded_single = [(-(-d // block_size) * block_size) if not diag else d for d, diag in zip(single, dim_diag)]
        stack = int(max(1, np.prod([max(1, len(s)) for s in split_sizes])))
        self._padded_stacked_shape = tuple([stack] + padded_single)

    def partition(self, x: jax.Array) -> Tuple[jax.Array, ...]:
        assert tuple(x.shape) == self._shape
        tensors = [x]
        for axis, indices in self._splits:
            nxt = []
            for t in tensors:
                nxt.extend(jnp.split(t, indices, axis=axis))
            tensors = nxt
        return tuple(tensors)

    def merge_partitions(self, parts: List[jax.Array]) -> jax.Array:
        for axis, indices in reversed(self._splits):
            n = len(indices) + 1
            merged, k = [], 0
            while k < len(parts):
                merged.append(jnp.concatenate(parts[k : k + n], axis=axis))
                k += n
            parts = merged
        assert len(parts) == 1
        return parts[0]

    def partition_shapes(self) -> List[Tuple[int, ...]]:
        shapes = [tuple(self._shape)]
        for axis, sizes in enumerate(self._split_sizes):
            new = []
            for base in shapes:
                for s in sizes.tolist():
                    b = list(base)
                    b[axis] = int(s)
                    new.append(tuple(b))
            shapes = new
        return shapes


def _pad_and_stack_matrices(arrs: List[jax.Array], block: int, pad_mask: Optional[List[bool]] = None):
    is_scalar = arrs[0].ndim == 0
    items = [a[None] if is_scalar else a for a in arrs]
    shapes = [a.shape for a in items]
    max_dims = [max(s[i] for s in shapes) for i in range(len(shapes[0]))]
    pad_mask = [True] * len(max_dims) if pad_mask is None else pad_mask
    target = [(-(-d // block) * block) if pad_mask[i] else d for i, d in enumerate(max_dims)]
    padded = [jnp.pad(a, [(0, target[i] - a.shape[i]) for i in range(a.ndim)]) for a in items]
    return jnp.stack(padded)


def _unstack_and_unpad_matrices(stacked: jax.Array, original_shapes: List[Tuple[int, ...]]):
    is_scalar = len(original_shapes[0]) == 0
    pieces = jnp.split(stacked, stacked.shape[0], 0)
    out = []
    for arr, orig in zip(pieces, original_shapes):
        arr = jnp.squeeze(arr, 0)
        if is_scalar:
            out.append(arr[0])
        else:
            out.append(arr[tuple(slice(0, d) for d in orig)])
    return tuple(out)


def _init_Q_exprs(
    t_shape: Tuple[int, ...],
    original_shape: Tuple[int, ...],
    scale: float,
    dim_diag: List[bool],
    q_dtype: jnp.dtype,
    exprs_only: bool = False,
):
    """initialize per-axis Q and L or just einsum expressions for QUAD updates.

    when `exprs_only=True`, returns only (exprP, exprGs) strings needed by einsum.
    otherwise returns (Q_list, L_list, (exprP, exprGs)).
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(t_shape) > 13:
        raise ValueError(">13 dims not supported by einsum letter space.")
    if len(t_shape) == 0:
        t_shape, original_shape, dim_diag = (1,), (1,), [True]

    scale = scale ** (1 / max(1, len(t_shape)))
    Q, L = [], []
    p1, p2, p3, p4, exprGs = [], [], "", "", []

    for i, (size, o_size, is_diag) in enumerate(zip(t_shape, original_shape, dim_diag)):
        if not exprs_only:
            L.append(jnp.zeros((1,), dtype=jnp.float32))
        if is_diag:
            if not exprs_only:
                q = scale * jnp.ones(o_size, dtype=q_dtype)
                q = q[:size] if size <= o_size else jnp.pad(q, (0, size - o_size))
                Q.append(q)
            b = letters[i + 13]
            p1.append(b)
            p2.append(b)
            p3 += b
            p4 += b
            sub = "".join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{sub},{sub}->{b}")
        else:
            if not exprs_only:
                q = scale * jnp.eye(o_size, dtype=q_dtype)
                q = q[:size, :size] if size <= o_size else jnp.pad(q, ((0, size - o_size), (0, size - o_size)))
                Q.append(q)
            a, b, c = letters[i], letters[i + 13], letters[i + 26]
            p1.append(a + b)
            p2.append(a + c)
            p3 += c
            p4 += b
            s1 = "".join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            s2 = "".join(letters[i + 26] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{s1},{s2}->{b}{c}")

    exprP = ",".join(p1) + "," + ",".join(p2) + "," + p3 + "->" + p4
    return (exprP, tuple(exprGs)) if exprs_only else (Q, L, (exprP, tuple(exprGs)))


def _norm_lower_bound(A: jax.Array):
    """fast lower bound on spectral norm using a power-iteration style pass."""
    scale = jnp.max(jnp.diag(A))
    A = A / scale
    j = jnp.argmax(jnp.sum(A * A, axis=1))
    x = A @ jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
    x = x / jnp.linalg.norm(x)
    return jnp.linalg.norm(A @ x) * scale


def _update_precond(Q_list, L_list, G, key, exprs, precond_lr, original_axis_sizes):
    # precondition gradient and build stats
    beta = 0.95
    max_rms = 1.1
    exprP, exprGs = exprs
    Pg = jnp.einsum(exprP, *Q_list, *Q_list, G + jax.random.normal(key, G.shape, G.dtype) * 1e-8)
    Pg_out = jnp.einsum(exprP, *Q_list, *Q_list, G)
    Pg_out /= jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(Pg_out))) / max_rms, 1.0)
    o_sizes = jnp.asarray(original_axis_sizes, dtype=jnp.float32)
    total = jnp.prod(o_sizes)

    # diagonal vs dense path
    def step(i, q, l, d_i):
        t1 = jnp.einsum(exprGs[i], Pg, Pg)
        if q.ndim < 2:  # diagonal
            t2 = total / d_i
            ell = jnp.max(t1) + t2
            l_new = jnp.maximum(beta * l + (1 - beta) * ell, ell)
            z = (precond_lr / (2 * l_new)).astype(q.dtype)
            gain = 1 - z * (t1 - t2)
            return q * gain * gain, l_new
        t2 = total / d_i
        ell = _norm_lower_bound(t1) + t2
        l_new = jnp.maximum(beta * l + (1 - beta) * ell, ell)
        z = (precond_lr / (2 * l_new)).astype(q.dtype)
        p = q - z * (t1 @ q - t2 * q)
        p = p - z * (p @ t1 - p * t2)
        return (p + p.T) / 2, l_new

    # aggregate updates and clamp rms
    outs = [step(i, q, l, o_sizes[i]) for i, (q, l) in enumerate(zip(Q_list, L_list))]
    Qn = [o[0] for o in outs]
    Ln = [o[1] for o in outs]
    return Qn, Ln, Pg_out


def _build_exprs(prepared_group, merged_diags_group, merged_shapes_group, q_dtype):
    return jax.tree.map(
        lambda g, dd, ms: (
            None if g is None else _init_Q_exprs(tuple(g.shape[1:]), tuple(ms), 1.0, list(dd), q_dtype, exprs_only=True)
        ),
        prepared_group,
        merged_diags_group,
        merged_shapes_group,
        is_leaf=lambda x: x is None,
    )


def _prepare(tree, group_mask, block_size: int, max_dense: int):
    """reshape, partition, pad, and stack per-leaf tensors for QUAD updates."""
    # filter to group and record shapes
    tree = jax.tree.map(lambda u, m: u if m else None, tree, group_mask)
    orig_shapes = jax.tree.map(lambda g: None if g is None else g.shape[1:], tree, is_leaf=lambda x: x is None)

    # merge dims and flag diag axes
    merged_shapes = jax.tree.map(
        lambda g: None if g is None else _merge_dims_and_diag(g.shape[1:], max_dense)[0],
        tree,
        is_leaf=lambda x: x is None,
    )
    merged_diags = jax.tree.map(
        lambda g: None if g is None else _merge_dims_and_diag(g.shape[1:], max_dense)[1],
        tree,
        is_leaf=lambda x: x is None,
    )
    # build partitioners and padded stacked shapes
    parts = jax.tree.map(
        lambda ms, dd: None if ms is None else BlockPartitioner(tuple(ms), block_size, list(dd)),
        merged_shapes,
        merged_diags,
        is_leaf=lambda x: x is None or isinstance(x, (list, tuple)),
    )
    part_shapes = jax.tree.map(
        lambda p: None if p is None else p._padded_stacked_shape, parts, is_leaf=lambda x: x is None
    )
    part_orig_shapes = jax.tree.map(
        lambda p: None if p is None else p.partition_shapes(), parts, is_leaf=lambda x: x is None
    )

    # reshape each batch sample to merged shapes
    prepared = jax.tree.map(
        lambda g, ms: None if g is None else vmap(lambda x: jnp.reshape(x, ms))(g),
        tree,
        merged_shapes,
        is_leaf=lambda x: x is None or isinstance(x, (list, tuple)),
    )

    def _part_stack(g, p, dd):
        g = jnp.reshape(g, p._shape)
        pad_mask = [not d for d in dd]
        return _pad_and_stack_matrices(list(p.partition(g)), block_size, pad_mask)

    # partition along non-diag axes, pad dense axes to blocks, and stack
    prepared = jax.tree.map(
        lambda g, p, dd: None if g is None else vmap(_part_stack, in_axes=(0, None, None))(g, p, dd),
        prepared,
        parts,
        merged_diags,
        is_leaf=lambda x: x is None,
    )

    def _merge_batch_stack(g):
        if g is None or g.ndim < 2:
            return g
        return g.reshape((g.shape[0] * g.shape[1],) + g.shape[2:])

    # merge [batch, stack] into a single leading dimension
    prepared = jax.tree.map(
        lambda g: None if g is None else _merge_batch_stack(g), prepared, is_leaf=lambda x: x is None
    )
    return prepared, orig_shapes, merged_shapes, parts, part_shapes, part_orig_shapes, merged_diags


def _unprepare(precond_gs, orig_shapes, part_shapes, parts, part_orig_shapes, scanned_layers_):
    """invert `_prepare`: unstack, unpad, merge partitions, reshape, drop dummy."""
    # split leading dimension back to [batch, stack]
    def _split_leading(g, ps):
        if g is None or ps is None:
            return g
        stack = ps[0]
        batch = g.shape[0] // stack
        return g.reshape((batch, stack) + g.shape[1:])

    precond_gs = jax.tree.map(
        lambda g, ps: None if g is None else _split_leading(g, ps), precond_gs, part_shapes, is_leaf=lambda x: x is None
    )
    # unstack/unpad partitions, merge back, and restore merged shapes
    def _unprep_leaf(g, os, p, p_shapes):
        g = vmap(lambda x: _unstack_and_unpad_matrices(x, p_shapes))(g)
        g = vmap(p.merge_partitions)(g)
        return vmap(lambda x: jnp.reshape(x, os))(g)

    precond_gs = jax.tree.map(
        lambda g, os, p, psh: None if g is None else _unprep_leaf(g, os, p, psh),
        precond_gs,
        orig_shapes,
        parts,
        part_orig_shapes,
        is_leaf=lambda x: x is None,
    )
    # drop dummy scan batch dimension for non-scanned layers
    precond_gs = jax.tree.map(
        lambda g, s: None if g is None else (g if s else jnp.squeeze(g, 0)),
        precond_gs,
        scanned_layers_,
        is_leaf=lambda x: x is None,
    )
    return precond_gs


def scale_by_quad(
    lr_style: Optional[str] = "adam",
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: Union[str, jnp.dtype] = jnp.bfloat16,
    scanned_layers: Optional[base.Params] = None,
    block_size: int = 512,
    pipeline_axis_name: Optional[str] = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: Optional[PartitionSpecTree] = None,
    **_: Any,
) -> base.GradientTransformation:
    dtype = canonicalize_dtype(dtype)
    assert dtype in (jnp.bfloat16, jnp.float32), "dtype must be bfloat16 or float32"

    def init_fn(params):
        # unbox flax partitioned params (if present)
        if HAVE_FLAX:
            params_unboxed = jax.tree.map(
                lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
                params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )
        else:
            params_unboxed = params

        # initialize momentum buffers (optional)
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=dtype), params_unboxed)
            if params_partition_specs is not None:
                mu = _shard(mu, params_partition_specs)

        # add dummy leading batch for unscanned layers
        scanned_layers_ = (
            scanned_layers if scanned_layers is not None else jax.tree.map(lambda _: False, params_unboxed)
        )
        params1 = jax.tree.map(lambda p, s: p if s else p[None], params_unboxed, scanned_layers_)

        # determine grouping masks: one_d vs large vs dense
        shapes_wo_lead = jax.tree.map(lambda x: x.shape[1:], params1)
        is_1d = jax.tree.map(_is_1d_like, shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple))
        has_large = jax.tree.map(
            lambda s: any(int(d) > max_size_dense for d in s), shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple)
        )
        is_dense = jax.tree.map(lambda a, b: (not a) and (not b), is_1d, has_large)

        def _init_group(mask):
            # prepare group and init q/l
            prepared, _os, m_shapes, _parts, p_shapes, _p_os, m_diags = _prepare(
                params1, mask, block_size, max_size_dense
            )
            triplets = jax.tree.map(
                lambda ps, dd, ms: (
                    None
                    if ps is None
                    else list(_init_Q_exprs(tuple(ps[1:]), tuple(ms), preconditioner_init_scale, list(dd), dtype))
                ),
                p_shapes,
                m_diags,
                m_shapes,
                is_leaf=lambda x: x is None or isinstance(x, (list, tuple)),
            )
            Qs, Ls, _ = [jax.tree.map(lambda _, t: None if t is None else t[i], prepared, triplets) for i in range(3)]

            def _broadcast(g, seq):  # replicate q/l across [B*stack]
                if seq is None:
                    return None
                lead = g.shape[0]
                return [jnp.broadcast_to(a, (lead,) + a.shape) for a in seq]

            # broadcast q/l across leading [B*stack]
            Qs = jax.tree.map(_broadcast, prepared, Qs, is_leaf=lambda x: x is None)
            Ls = jax.tree.map(_broadcast, prepared, Ls, is_leaf=lambda x: x is None)
            return Qs, Ls, p_shapes

        Qs_d, Ls_d, _ = _init_group(is_dense)
        Qs_ld, Ls_ld, _ = _init_group(has_large)
        Qs_1d, Ls_1d, _ = _init_group(is_1d)

        def _concat_qs_ls(qtree, ltree):
            # concat per-leaf q/l across the dense group
            flat_qs, _ = jax.tree.flatten(qtree, is_leaf=lambda x: x is None or isinstance(x, list))
            flat_qs = [q for q in flat_qs if q is not None]
            if not flat_qs:
                return None, None, []
            num = len(flat_qs[0])
            assert all(len(q) == num for q in flat_qs)
            split = [q[0].shape[0] for q in flat_qs]
            catQ = [jnp.concatenate([q[i] for q in flat_qs], 0) for i in range(num)]
            flat_ls, _ = jax.tree.flatten(ltree, is_leaf=lambda x: x is None or isinstance(x, list))
            flat_ls = [l for l in flat_ls if l is not None]
            catL = [jnp.concatenate([l[i] for l in flat_ls], 0) for i in range(len(flat_ls[0]))]
            return catQ, catL, split

        dense_Qs_cat, dense_Ls_cat, _ = _concat_qs_ls(Qs_d, Ls_d)

        # shard dense path if requested
        if dense_Qs_cat is not None and pipeline_axis_name is not None:
            dense_Qs_cat = _pad_shard_Q_axes_list(dense_Qs_cat, pipeline_axis_name, pipeline_axis_size)
            dense_Ls_cat = [_pad_shard_leading(l, pipeline_axis_name, pipeline_axis_size) for l in dense_Ls_cat]

        # shard large path per-leaf if requested
        if pipeline_axis_name is not None:
            Qs_ld = jax.tree.map(
                lambda q: None if q is None else _pad_shard_Q_axes_list(q, pipeline_axis_name, pipeline_axis_size),
                Qs_ld,
                is_leaf=lambda x: x is None or isinstance(x, list),
            )
            Ls_ld = jax.tree.map(
                lambda l: None if l is None else [_pad_shard_leading(x, pipeline_axis_name, pipeline_axis_size) for x in l],
                Ls_ld,
                is_leaf=lambda x: x is None or isinstance(x, list),
            )

        # build optimizer state
        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs=dict(dense_concat=dense_Qs_cat, large=Qs_ld, one_d=Qs_1d),
            Ls=dict(dense_concat=dense_Ls_cat, large=Ls_ld, one_d=Ls_1d),
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params | None = None):
        # compute step index and preconditioner lr
        step = safe_int32_increment(state["count"])
        plr = jnp.maximum(preconditioner_lr * jax.lax.rsqrt(1.0 + step / 10000.0), 0.4)

        # unbox flax partitioned updates if present
        flax_partitioned = False
        if HAVE_FLAX:
            flat, tdef = jax.tree.flatten(
                updates, is_leaf=lambda g: isinstance(g, (nn.Partitioned, jax.ShapeDtypeStruct))
            )
            if any(isinstance(g, nn.Partitioned) for g in flat):
                updates = tdef.unflatten([g.unbox() if isinstance(g, nn.Partitioned) else g for g in flat])
                flax_partitioned = True

        # momentum update in original shapes/specs (optional), cast, normalize
        mu = state["mu"]
        mupd = updates
        if mu is not None:
            mu = otu.tree_update_moment(updates, mu, b1, 1)
            if params_partition_specs is not None:
                mu = _shard(mu, params_partition_specs)
            mupd = otu.tree_bias_correction(mu, b1, step)
        mu = otu.tree_cast(mu, dtype)
        mupd = otu.tree_cast(mupd, dtype)
        if normalize_grads:
            mupd = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-6), mupd)

        # ensure leading batch for unscanned and recompute masks
        orig_shapes = jax.tree.map(lambda g: g.shape, mupd)
        scanned_layers_ = scanned_layers if scanned_layers is not None else jax.tree.map(lambda _: False, mupd)
        mupd1 = jax.tree.map(lambda g, s: g if s else g[None], mupd, scanned_layers_)

        shapes_wo_lead = jax.tree.map(lambda x: x.shape[1:], mupd1)
        is_1d = jax.tree.map(_is_1d_like, shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple))
        has_large = jax.tree.map(
            lambda s: any(int(d) > max_size_dense for d in s), shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple)
        )
        is_dense = jax.tree.map(lambda a, b: (not a) and (not b), is_1d, has_large)

        # dense path: prepare, concat, optional shard, update, scatter back
        dense_prep, d_os, d_mshapes, d_parts, d_pshapes, d_porig, d_diags = _prepare(
            mupd1, is_dense, block_size, max_size_dense
        )
        flat, tdef = jax.tree.flatten(dense_prep)
        dense_leaves = [x for x in flat if x is not None]
        if dense_leaves:
            splits = [x.shape[0] for x in dense_leaves]
            grads_cat = jnp.concatenate(dense_leaves, 0)
            # build per-sample original axis sizes by repeating per-leaf merged shapes
            flat_ms, _ = jax.tree.flatten(d_mshapes, is_leaf=lambda x: isinstance(x, (list, tuple)))
            ms_leaves = [ms for x, ms in zip(flat, flat_ms) if x is not None]
            pieces = []
            for ms, n in zip(ms_leaves, splits):
                oz = jnp.asarray(ms, jnp.int32)
                pieces.append(jnp.broadcast_to(oz, (n, oz.shape[0])))
            axis_sizes_cat = jnp.concatenate(pieces, 0) if pieces else None
            exprs_dense = _init_Q_exprs(
                tuple(grads_cat.shape[1:]),
                tuple(grads_cat.shape[1:]),
                1.0,
                [False] * (grads_cat.ndim - 1),
                dtype,
                exprs_only=True,
            )
            Qcat, Lcat = state["Qs"]["dense_concat"], state["Ls"]["dense_concat"]
            if Qcat is None:
                raise ValueError("Found dense gradients but no optimizer state. This indicates an inconsistency.")
            else:
                # pad grads to match state rows; optional pipeline sharding
                need = Qcat[0].shape[0]
                pad = need - grads_cat.shape[0]
                if pad < 0:
                    raise ValueError("Dense concat Q smaller than grads; check block_size/grouping.")
                grads_cat = grads_cat if pad == 0 else jnp.pad(grads_cat, [(0, pad)] + [(0, 0)] * (grads_cat.ndim - 1))
                if axis_sizes_cat is not None:
                    axis_sizes_cat = (
                        axis_sizes_cat
                        if pad == 0
                        else jnp.pad(axis_sizes_cat, [(0, pad), (0, 0)], constant_values=1)
                    )
                if pipeline_axis_name is not None:
                    grads_cat = _shard(grads_cat, PartitionSpec(pipeline_axis_name))
                    if axis_sizes_cat is not None:
                        axis_sizes_cat = _shard(axis_sizes_cat, PartitionSpec(pipeline_axis_name))

                key = jax.random.fold_in(jax.random.PRNGKey(42), step)
                keys = jax.random.split(key, grads_cat.shape[0])

                # generic vmap over list-pytrees of q/l
                Qnew, Lnew, Pg_cat = vmap(
                    lambda Qs, Ls, g, k, oz: _update_precond(Qs, Ls, g, k, exprs_dense, plr, oz)
                )(Qcat, Lcat, grads_cat, keys, axis_sizes_cat)

                valid = sum(splits)

                # persist q/l and scatter grads back to tree
                Qcat = [otu.tree_cast(Qnew[i], dtype) for i in range(len(Qcat))]
                Lcat = [otu.tree_cast(Lnew[i], jnp.float32) for i in range(len(Lcat))]
                state["Qs"]["dense_concat"], state["Ls"]["dense_concat"] = Qcat, Lcat

                Pg_cat = Pg_cat[:valid]
                flat_out, start = [], 0
                for x in flat:
                    if x is None:
                        flat_out.append(None)
                    else:
                        sz = x.shape[0]
                        flat_out.append(Pg_cat[start : start + sz])
                        start += sz
                dense_tree = tdef.unflatten(flat_out)

        else:
            dense_tree = jax.tree.map(lambda _: None, dense_prep)

        # large path: prepare per-leaf, optional shard, update, slice valid
        large_prep, l_os, l_mshapes, l_parts, l_pshapes, l_porig, l_diags = _prepare(
            mupd1, has_large, block_size, max_size_dense
        )
        if pipeline_axis_name is not None:
            large_valid = jax.tree.map(
                lambda g: None if g is None else g.shape[0], large_prep, is_leaf=lambda x: x is None
            )
            large_prep = jax.tree.map(
                lambda g: _pad_shard_leading(g, pipeline_axis_name, pipeline_axis_size),
                large_prep,
                is_leaf=lambda x: x is None,
            )
        exprs_large = _build_exprs(large_prep, l_diags, l_mshapes, dtype)

        key = jax.random.fold_in(jax.random.PRNGKey(42), step)

        def _vmapped_update(g, Qlst, Llst, ex, osz):
            if g is None:
                return None, None, None
            keys = jax.random.split(key, g.shape[0])
            oz = jnp.asarray(osz, jnp.int32)
            oz_mat = jnp.broadcast_to(oz, (g.shape[0], oz.shape[0]))
            if pipeline_axis_name is not None:
                oz_mat = _shard(oz_mat, PartitionSpec(pipeline_axis_name))
            return vmap(lambda Qs, Ls, gg, kk, oo: _update_precond(Qs, Ls, gg, kk, ex, plr, oo))(
                Qlst, Llst, g, keys, oz_mat
            )

        QsL = jax.tree.map(
            _vmapped_update,
            large_prep,
            state["Qs"]["large"],
            state["Ls"]["large"],
            exprs_large,
            l_mshapes,
            is_leaf=lambda x: x is None or isinstance(x, (list, tuple)),
        )
        # keep state structure stable — take only q/l updates
        Qs_new = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[0] is None) else [otu.tree_cast(q, dtype) for q in t[0]],
            QsL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        Ls_new = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[1] is None) else [otu.tree_cast(l, jnp.float32) for l in t[1]],
            QsL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        precond_large = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[2] is None) else t[2],
            QsL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        # slice back to valid rows to discard padding
        precond_large = jax.tree.map(
            lambda g, v: None if (g is None or v is None) else g[:v],
            precond_large,
            large_valid,
            is_leaf=lambda x: x is None,
        )
        state["Qs"]["large"], state["Ls"]["large"] = Qs_new, Ls_new

        # one_d path: diag-only update, replicated
        one_prep, o_os, o_mshapes, o_parts, o_pshapes, o_porig, o_diags = _prepare(
            mupd1, is_1d, block_size, max_size_dense
        )
        exprs_one = _build_exprs(one_prep, o_diags, o_mshapes, dtype)
        Qs1 = jax.tree.map(
            lambda q: None if q is None else [q[0]],
            state["Qs"]["one_d"],
            is_leaf=lambda x: x is None or isinstance(x, list),
        )
        Ls1 = jax.tree.map(
            lambda l: None if l is None else [l[0]],
            state["Ls"]["one_d"],
            is_leaf=lambda x: x is None or isinstance(x, list),
        )
        oneL = jax.tree.map(
            _vmapped_update,
            one_prep,
            Qs1,
            Ls1,
            exprs_one,
            o_mshapes,
            is_leaf=lambda x: x is None or isinstance(x, (list, tuple)),
        )
        Qs1n = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[0] is None) else [otu.tree_cast(q, dtype) for q in t[0]],
            oneL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        Ls1n = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[1] is None) else [otu.tree_cast(l, jnp.float32) for l in t[1]],
            oneL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        precond_one = jax.tree_util.tree_map(
            lambda t: None if (t is None or t[2] is None) else t[2],
            oneL,
            is_leaf=lambda x: isinstance(x, tuple) or x is None,
        )
        state["Qs"]["one_d"], state["Ls"]["one_d"] = Qs1n, Ls1n

        # unprepare each group and merge results
        dense_unp = _unprepare(dense_tree, d_os, d_pshapes, d_parts, d_porig, scanned_layers_)
        large_unp = _unprepare(precond_large, l_os, l_pshapes, l_parts, l_porig, scanned_layers_)
        one_unp = _unprepare(precond_one, o_os, o_pshapes, o_parts, o_porig, scanned_layers_)
        precond_all = jax.tree.map(
            lambda d, l, o: d if d is not None else (l if l is not None else o),
            dense_unp,
            large_unp,
            one_unp,
            is_leaf=lambda x: x is None,
        )

        # restore original param shapes/specs and optional scaling
        precond_all = jax.tree.map(
            lambda g, s: None if g is None else jnp.reshape(g, s), precond_all, orig_shapes, is_leaf=lambda x: x is None
        )
        if params_partition_specs is not None:
            precond_all = _shard(precond_all, params_partition_specs)

        if lr_style == "adam":
            precond_all = jax.tree.map(lambda g: g / 5.0, precond_all)

        # re-box flax containers if we unboxed at entry
        if flax_partitioned:
            flat_p, tdef_p = jax.tree.flatten(
                params, is_leaf=lambda g: hasattr(g, "unbox") or isinstance(g, jax.ShapeDtypeStruct)
            )
            if any(hasattr(p, "replace_boxed") for p in flat_p):
                flat_g, _ = jax.tree.flatten(precond_all)
                precond_all = tdef_p.unflatten(
                    [p.replace_boxed(g) if hasattr(p, "replace_boxed") else g for p, g in zip(flat_p, flat_g)]
                )

        # persist step counter and momentum
        state["count"] = step
        state["mu"] = otu.tree_cast(mu, dtype)
        return precond_all, state

    return base.GradientTransformation(init_fn, update_fn)


def quad(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    lr_style: Optional[str] = "adam",
    b1: float = 0.95,
    weight_decay: float = 0.1,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    preconditioner_lr: float = 0.7,
    preconditioner_init_scale: float = 1.0,
    dtype: Union[str, jnp.dtype] = jnp.bfloat16,
    scanned_layers: Optional[base.Params] = None,
    block_size: int = 512,
    pipeline_axis_name: Optional[str] = None,
    pipeline_axis_size: int = 1,
    params_partition_specs: Optional[PartitionSpecTree] = None,
) -> base.GradientTransformation:
    tx = [
        scale_by_quad(
            lr_style=lr_style,
            b1=b1,
            normalize_grads=normalize_grads,
            max_size_dense=max_size_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            dtype=dtype,
            scanned_layers=scanned_layers,
            block_size=block_size,
            pipeline_axis_name=pipeline_axis_name,
            pipeline_axis_size=pipeline_axis_size,
            params_partition_specs=params_partition_specs,
        )
    ]
    if weight_decay > 0.0:
        tx.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    tx.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*tx)


def get_opt_state_partition_specs(params, **quad_kwargs):
    """return pytree of PartitionSpec (or None) for optimizer state, no arrays allocated."""
    # extract inputs and defaults
    b1 = quad_kwargs.get("b1", 0.95)
    scanned_layers = quad_kwargs.get("scanned_layers", None)
    max_size_dense = quad_kwargs.get("max_size_dense", 8192)
    pipeline_axis_name = quad_kwargs.get("pipeline_axis_name", None)
    params_partition_specs = quad_kwargs.get("params_partition_specs", None)
    weight_decay = quad_kwargs.get("weight_decay", 0.0)

    # unbox flax partitioned params if present
    if HAVE_FLAX:
        params_unboxed = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )
    else:
        params_unboxed = params

    # grouping masks using merged shapes semantics
    scanned_layers_ = scanned_layers if scanned_layers is not None else jax.tree.map(lambda _: False, params_unboxed)
    shapes_wo_lead = jax.tree.map(lambda p, s: p.shape[1:] if s else p.shape, params_unboxed, scanned_layers_)
    is_1d = jax.tree.map(_is_1d_like, shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple))
    has_large = jax.tree.map(
        lambda s: any(int(d) > max_size_dense for d in s), shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple)
    )
    is_dense = jax.tree.map(lambda a, b: (not a) and (not b), is_1d, has_large)

    # helper: count axes after merging dims
    def _num_axes_after_merge(shape_tup):
        return len(_merge_dims_and_diag(tuple(map(int, shape_tup)), max_size_dense)[0])

    replicated = PartitionSpec()

    # dense-group specs — ensure consistent merged axis count
    flat_shapes, _ = jax.tree.flatten(shapes_wo_lead, is_leaf=lambda x: isinstance(x, tuple))
    flat_dense, _ = jax.tree.flatten(is_dense, is_leaf=lambda x: isinstance(x, bool))
    dense_indices = [i for i, m in enumerate(flat_dense) if m]
    if dense_indices:
        n_axes_dense = _num_axes_after_merge(flat_shapes[dense_indices[0]])
        for idx in dense_indices[1:]:
            if _num_axes_after_merge(flat_shapes[idx]) != n_axes_dense:
                raise ValueError("inconsistent dense merged axis counts")
        dense_Qs_concat_specs = [
            PartitionSpec(pipeline_axis_name) if pipeline_axis_name is not None else replicated
        ] * n_axes_dense
        dense_Ls_concat_specs = [
            PartitionSpec(pipeline_axis_name) if pipeline_axis_name is not None else replicated
        ] * n_axes_dense
    else:
        dense_Qs_concat_specs = None
        dense_Ls_concat_specs = None

    # large/one_d specs per leaf (use pipeline axis where applicable)
    large_Q_specs = jax.tree.map(
        lambda shape, mask: (
            None
            if not mask
            else [PartitionSpec(pipeline_axis_name) if pipeline_axis_name is not None else replicated]
            * _num_axes_after_merge(shape)
        ),
        shapes_wo_lead,
        has_large,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    large_L_specs = jax.tree.map(
        lambda shape, mask: (
            None
            if not mask
            else [PartitionSpec(pipeline_axis_name) if pipeline_axis_name is not None else replicated]
            * _num_axes_after_merge(shape)
        ),
        shapes_wo_lead,
        has_large,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    one_d_Q_specs = jax.tree.map(lambda m: None if not m else [replicated], is_1d)
    one_d_L_specs = jax.tree.map(lambda m: None if not m else [replicated], is_1d)

    # momentum specs — default to param sharding if not provided
    mu_specs = None
    if b1 > 0:
        if params_partition_specs is not None:
            mu_specs = params_partition_specs
        else:

            def _param_spec(p):
                try:
                    return p.sharding.spec
                except Exception:
                    return replicated

            mu_specs = jax.tree.map(_param_spec, params_unboxed)

    # return optimizer-state partitionspec structure (matches state layout)
    precond_specs = dict(
        count=replicated,
        mu=mu_specs,
        Qs=dict(dense_concat=dense_Qs_concat_specs, large=large_Q_specs, one_d=one_d_Q_specs),
        Ls=dict(dense_concat=dense_Ls_concat_specs, large=large_L_specs, one_d=one_d_L_specs),
    )
    if weight_decay and weight_decay > 0:
        return (precond_specs, None, None)
    else:
        return (precond_specs, None)
