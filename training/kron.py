from typing import Any, List, Optional, Union, Callable, Tuple
from functools import partial
import string
from collections import defaultdict
import numpy as np

import chex
import jax
from jax import vmap
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding as NS, PartitionSpec as P
import flax.linen as nn
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 250 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.minimum(
            jnp.maximum(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob),
            max_prob,
        )

    return _schedule


def scale_by_kron(
    b1: float = 0.9,
    normalize_grads: bool = False,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    max_merged_dim_size: int = 4096,
    partition_grads_into_blocks: bool = False,
    block_size: int = 128,
    mesh: Optional[jax.sharding.Mesh] = None,
    axis_name: str = "data",
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        b1: float, momentum parameter.
        normalize_grads: bool, whether to normalize the incoming gradients.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge dimensions of tensors with
            more than 2 dimensions to improve compile times and preconditioner
            efficacy.
        max_merged_dim_size: int, max product of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for memory efficiency.
        block_size: int, size of partitions to use for memory efficiency.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)
    preconditioner_lr = 0.1
    preconditioner_init_scale = 1.0
    lax_map = lax_map_scanned_layers
    bs = lax_map_batch_size

    def init_fn(params):
        params = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda v: isinstance(v, (chex.Array, nn.Partitioned)),
        )

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        scanned_sizes = jax.tree.map(
            lambda p, s: p.shape[0] if s else 0, params, scanned_layers_
        )

        # momentum
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)

        params, params_struct = jax.tree.flatten(params)
        scanned_layers_ = params_struct.flatten_up_to(scanned_layers_)
        scanned_sizes = params_struct.flatten_up_to(scanned_sizes)
        params_without_scan = [
            _first_n_dims(p, int(s)) for p, s in zip(params, scanned_layers_)
        ]

        # merge dimensions
        if merge_small_dims:
            merged_shapes = [
                _merge_small_dims(p.shape, max_merged_dim_size)
                for p in params_without_scan
            ]
            params_without_scan = [
                jnp.reshape(p, ns) for p, ns in zip(params_without_scan, merged_shapes)
            ]

        # partition grads into blocks
        if partition_grads_into_blocks:
            partitioners = [
                BlockPartitioner(
                    p.shape,
                    block_size,
                    max_size_triangular,
                    min_ndim_triangular,
                    memory_save_mode,
                )
                for p in params_without_scan
            ]
            # this becomes list of tuples where each tuple contains layer's partitions
            params_without_scan = [
                p_cls.partition(p)
                for p_cls, p in zip(partitioners, params_without_scan)
            ]
            # stack same-shaped partitions per layer
            params_without_scan = [_stack_matrices(p) for p in params_without_scan]

        # initialize preconditioners
        Qs = [
            jax.tree.map(
                lambda t: _init_Q_exprs(
                    _first_n_dims(t, int(partition_grads_into_blocks)),
                    preconditioner_init_scale,
                    max_size_triangular,
                    min_ndim_triangular,
                    memory_save_mode,
                    precond_dtype,
                )[0],
                p,
            )
            for p in params_without_scan
        ]
        # Qs is now list (params level) of tuples (partitions level) of lists (Qs)
        # broadcast for scans and stacks
        all_Qs = []
        for q, p, s in zip(Qs, params_without_scan, scanned_sizes):
            # first add leading dim for stacked partitions
            new_q = q
            if partition_grads_into_blocks:
                partitions = []
                for part_qs, part_p in zip(q, p):
                    partitions.append(
                        jax.tree.map(
                            lambda d: jnp.repeat(
                                jnp.expand_dims(d, 0), part_p.shape[0], axis=0
                            ),
                            part_qs,
                        )
                    )
                new_q = tuple(partitions)
            # then add a leading dim if we're scanning this layer
            if s > 0:
                new_q = jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), new_q
                )
            all_Qs.append(new_q)
        # for a stacked and scanned layer, it now has two leading dims to be vmapped
        Qs = all_Qs
        Qs = params_struct.unflatten(Qs)

        # Calculate sizes for nu (preconditioner) and mu (momentum)
        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            print(
                f"PSGD Preconditioners size: {Qs_n_elements} elements, "
                f"{Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        # initial state
        return dict(count=jnp.zeros([], jnp.int32), mu=mu, Qs_preconditioners=Qs)

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = jax.random.fold_in(jax.random.PRNGKey(5318008), state["count"])

        # account for flax.linen.Partitioned grads and params
        boxed_updates, grads_structure = jax.tree.flatten(
            updates, is_leaf=lambda v: isinstance(v, (chex.Array, nn.Partitioned))
        )
        flax_partitioned = False
        if isinstance(boxed_updates[0], nn.Partitioned):
            flax_partitioned = True
            updates = [u.unbox() for u in boxed_updates]
            updates = grads_structure.unflatten(updates)

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        # normalize grads
        def norm_grads(g):
            return g / (jnp.linalg.norm(g) + 1e-16)

        if normalize_grads:
            updates = jax.tree.map(norm_grads, updates)

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        momentum_updates = grads_structure.flatten_up_to(momentum_updates)
        Qs = grads_structure.flatten_up_to(state["Qs_preconditioners"])
        scanned_layers_ = grads_structure.flatten_up_to(scanned_layers_)

        # merge dimensions
        if merge_small_dims:
            original_shapes = [
                p.shape[1:] if s else p.shape
                for p, s in zip(momentum_updates, scanned_layers_)
            ]
            merged_shapes = [
                _merge_small_dims(os, max_merged_dim_size) for os in original_shapes
            ]
            momentum_updates = [
                _map_fn(False, 0, int(s), lambda p: jnp.reshape(p, ns), p)
                for s, p, ns in zip(scanned_layers_, momentum_updates, merged_shapes)
            ]

        # partition grads into blocks
        if partition_grads_into_blocks:
            partitioners = [
                BlockPartitioner(
                    p.shape[1:] if s else p.shape,
                    block_size,
                    max_size_triangular,
                    min_ndim_triangular,
                    memory_save_mode,
                )
                for s, p in zip(scanned_layers_, momentum_updates)
            ]
            # this becomes list of tuples where each tuple contains layer's partitions
            momentum_updates = [
                _map_fn(False, 0, int(s), p_cls.partition, p)
                for s, p, p_cls in zip(scanned_layers_, momentum_updates, partitioners)
            ]
            partitioned_shapes = [
                jax.tree.map(lambda x: x.shape[1:] if s else x.shape, p)
                for p, s in zip(momentum_updates, scanned_layers_)
            ]
            # stack same-shaped partitions per layer, becoming single stacked, double
            # stacked if scanned.
            momentum_updates = [
                _map_fn(False, 0, int(s), _stack_matrices, p)
                for s, p in zip(scanned_layers_, momentum_updates)
            ]
            revert_indices = [
                _sort_and_group_matrices(layers)[3] for layers in partitioned_shapes
            ]
            n_dims_to_map = [1 + int(s) for s in scanned_layers_]
        else:
            # put everything in a tuple for consistency
            momentum_updates = [(x,) for x in momentum_updates]
            Qs = [(x,) for x in Qs]
            n_dims_to_map = [int(s) for s in scanned_layers_]

        # get einsum expressions
        expressions = [
            tuple(
                _init_Q_exprs(
                    _first_n_dims(g, nm),
                    preconditioner_init_scale,
                    max_size_triangular,
                    min_ndim_triangular,
                    memory_save_mode,
                    precond_dtype,
                    existing_Q=jax.tree.map(lambda q: _first_n_dims(q, nm), Q),
                )
                for g, Q in zip(part_gs, part_Qs)
            )
            for part_gs, part_Qs, nm in zip(momentum_updates, Qs, n_dims_to_map)
        ]

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precond_update_precision):
                # create random vectors
                key, subkey = jax.random.split(key)
                Vs = _tree_random_like(subkey, momentum_updates)

                # balance preconditioners about every 100 updates
                def balance_Qs(Qs_to_bal):
                    def _balance_Q(Q):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.prod(norms) ** (1 / len(norms))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return [
                        tuple(
                            _map_fn(False, 0, nm, _balance_Q, Q) if len(Q) > 1 else Q
                            for Q in part_qs
                        )
                        for part_qs, nm in zip(Qs_to_bal, n_dims_to_map)
                    ]

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) <= 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)
                if mesh is not None:
                    Qs = _constrain(Qs, mesh, axis_name)

                # form conjB
                conjBs = [
                    tuple(
                        _map_fn(lax_map, bs, nm, _conjB, Q, g, v)
                        for Q, g, v in zip(part_qs, part_gs, part_vs)
                    )
                    for part_gs, part_qs, part_vs, nm in zip(
                        momentum_updates, Qs, Vs, n_dims_to_map
                    )
                ]

                # update Qs
                new_Qs = [
                    tuple(
                        _map_fn(
                            lax_map,
                            bs,
                            nm,
                            partial(
                                _update_precond,
                                exprs=exprs,
                                precond_lr=preconditioner_lr,
                            ),
                            Q,
                            g,
                            conjb,
                        )
                        for exprs, g, Q, conjb in zip(
                            part_exprs, part_gs, part_qs, part_bs
                        )
                    )
                    for part_exprs, part_gs, part_qs, part_bs, nm in zip(
                        expressions, momentum_updates, Qs, conjBs, n_dims_to_map
                    )
                ]
                if mesh is not None:
                    new_Qs = _constrain(new_Qs, mesh, axis_name)
                return new_Qs

        key, subkey = jax.random.split(key)
        do_update = jax.random.uniform(subkey, dtype=jnp.float32) < update_prob_in
        key, subkey = jax.random.split(key)
        Qs = jax.lax.cond(
            do_update, update_preconditioner, lambda _, qs: qs, subkey, Qs
        )

        # precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            precond_gs = [
                tuple(
                    _map_fn(lax_map, bs, nm, partial(_precond_grad, exprs=exprs), Q, g)
                    for exprs, Q, g in zip(part_exprs, part_qs, part_gs)
                )
                for part_exprs, part_gs, part_qs, nm in zip(
                    expressions, momentum_updates, Qs, n_dims_to_map
                )
            ]

        # unpartition grads
        if partition_grads_into_blocks:
            precond_gs = [
                _map_fn(False, 0, int(s), lambda p: _unstack_matrices(p, ri), p)
                for s, p, ri in zip(scanned_layers_, precond_gs, revert_indices)
            ]
            precond_gs = [
                _map_fn(False, 0, int(s), part.merge_partitions, p)
                for s, p, part in zip(scanned_layers_, precond_gs, partitioners)
            ]
        else:
            # pull everything out of their tuples
            precond_gs = [x[0] for x in precond_gs]
            Qs = [x[0] for x in Qs]

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = [
                _map_fn(False, 0, int(s), lambda p: jnp.reshape(p, os), p)
                for s, os, p in zip(scanned_layers_, original_shapes, precond_gs)
            ]

        # box preconditioned grads
        if flax_partitioned:
            precond_gs = [
                u.replace_boxed(pg) for u, pg in zip(boxed_updates, precond_gs)
            ]

        # unflatten pytrees
        new_updates = grads_structure.unflatten(precond_gs)
        Qs = grads_structure.unflatten(Qs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(count=count_inc, mu=mu, Qs_preconditioners=Qs)

        return new_updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    max_merged_dim_size: int = 4096,
    partition_grads_into_blocks: bool = False,
    block_size: int = 128,
    mesh: Optional[jax.sharding.Mesh] = None,
    axis_name: str = "data",
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        b1: float, momentum parameter.
        weight_decay: float, weight decay.
        weight_decay_mask: optional Any or callable, pytree of bool same structure
            as params with weight decay applied to True elements.
        normalize_grads: bool, whether to normalize the incoming gradients.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular. 'one_diag' sets only the largest
            or last dim in a layer to be diagonal, and 'all_diag' sets all preconditioners
            to be diagonal.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precond_update_precision: str, precision for matmul during preconditioner update,
            'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
            'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge dimensions of tensors with
            more than 2 dimensions to improve compile times and preconditioner
            efficacy.
        max_merged_dim_size: int, max product of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for memory efficiency.
        block_size: int, size of partitions to use for memory efficiency.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
            normalize_grads=normalize_grads,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            merge_small_dims=merge_small_dims,
            max_merged_dim_size=max_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            mesh=mesh,
            axis_name=axis_name,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def _get_preconditioner_types(
    shape: Tuple[int, ...],
    max_size: int,
    min_ndim_triangular: int,
    memory_save_mode: Optional[str],
) -> List[bool]:
    if len(shape) == 0:
        return True

    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        dim_diag[rev_sorted_dims[0]] = True
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
            "[None, 'one_diag', 'all_diag']"
        )

    for i, size in enumerate(shape):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular:
            dim_diag[i] = True

    return dim_diag


def _init_Q_exprs(
    t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype, existing_Q=None
):
    """For a scalar or tensor `t`, we initialize its preconditioner `Q` and
    reusable contraction expressions for updating `Q` and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )
        scale = scale ** (1 / len(shape))
        dim_diag = _get_preconditioner_types(
            shape, max_size, min_ndim_triangular, memory_save_mode
        )
        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.eye(size, dtype=dtype))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return exprA, exprGs, exprP
    return Q, (exprA, exprGs, exprP)


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and
    sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A). Looks to be a very
    tight lower bound.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        A_conj = A.conj()

        aa = jnp.real(A * A_conj)

        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x.conj() @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A_conj.T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x.conj()
            return max_abs * jnp.linalg.norm(A_conj.T @ (x / jnp.linalg.norm(x)))

        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    def no_calc(_):
        return max_abs

    return jax.lax.cond(max_abs > 0, calc, no_calc, A)


def _solve_triangular_right(X, A):
    """Compute X @ inv(A).

    A triangular solve has roughly the same complexity as a matmul.
    """
    X_ndim = X.ndim
    if X_ndim < 2:
        X = X[None, :]

    dtype_in = jnp.promote_types(A.dtype, X.dtype)
    A, X = A.astype(dtype_in), X.astype(dtype_in)
    leading_dims = 0
    if X.ndim > 2:
        leading_dims = X.ndim - 2
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=False, lower=False)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    solution = solve_fn(A, X)

    if X_ndim < 2:
        return solution[0]
    return solution


def _conjB(Q, G, V):
    """Compute conjB."""
    order = G.ndim
    p = list(range(order))
    conjB = jnp.transpose(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.ndim < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = jnp.swapaxes(conjB, i, order - 1)
    return conjB


def _update_precond(Q, G, conjB, exprs, precond_lr):
    """Compute A and update Q."""
    exprA, exprGs, _ = exprs

    A = jnp.einsum(exprA, *Q, G)

    A_conj = A.conj()
    conjB_conj = conjB.conj()

    def _update_single_q(i, q):
        term1 = jnp.einsum(exprGs[i], A, A_conj)
        term2 = jnp.einsum(exprGs[i], conjB_conj, conjB)

        if q.ndim < 2:
            q -= (
                precond_lr
                / _add_tiny(jnp.max(jnp.abs(term1 + term2)))
                * (term1 - term2)
                * q
            )
        else:
            q -= (
                precond_lr
                / _add_tiny(_norm_lower_bound(term1 + term2))
                * jnp.triu(term1 - term2)
                @ q
            )
        return q

    return [_update_single_q(i, q) for i, q in enumerate(Q)]


def _precond_grad(Q, G, exprs):
    """Precondition gradient G with preconditioner Q."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *[q.conj() for q in Q], *Q, G)


def _shard_psgd_precond(Q, axis_name):
    shape = Q.shape
    sharding = [None for _ in shape]
    if len(shape) < 2 or shape[-2] != shape[-1]:
        # probably a diagonal preconditioner
        return P(*sharding)
    # Shard if bigger than 0.1 MB
    if np.prod(shape) * Q.dtype.itemsize >= 0.1 * (2**20):
        sharding[-2] = axis_name
    return P(*sharding)


def _constrain(Qs, mesh, axis_name):
    return jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(
            x, NS(mesh, _shard_psgd_precond(x, axis_name))
        ),
        Qs,
    )


def _add_tiny(x):
    return x + jnp.finfo(x.dtype).tiny


def _first_n_dims(x, n):
    if n <= 0:
        return x
    indices = (0,) * n
    return x[indices + (slice(None),) * (x.ndim - n)]


def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Maybe map a fn along multiple leading axes."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        mapped_fn = lambda xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.lax.map(mapped_fn, xs=args, batch_size=bs if bs > 1 else None)
    else:
        mapped_fn = lambda *xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return vmap(mapped_fn)(*args)


def _tree_random_like(
    rng_key: chex.PRNGKey, target_tree: chex.ArrayTree
) -> chex.ArrayTree:
    tree_def = jax.tree.structure(target_tree)
    keys = jax.random.split(rng_key, tree_def.num_leaves)
    keys_tree = jax.tree.unflatten(tree_def, keys)
    return jax.tree.map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype), target_tree, keys_tree
    )


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    From distributed shampoo but slightly modified.
    """

    def __init__(
        self, param_shape, block_size, max_size, min_ndim_triangular, memory_save_mode
    ):
        # we don't split dimensions that will only have a diagonal preconditioner
        dim_diag = _get_preconditioner_types(
            param_shape, max_size, min_ndim_triangular, memory_save_mode
        )

        self._shape = param_shape
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d and not dim_diag[i]:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for i, indices in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tuple(tensors)

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for i, indices in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(
                    jnp.concatenate(partitions[ind : ind + n], axis=i)
                )
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def _merge_small_dims(shape_to_merge, max_dim) -> List[int]:
    """Merge small dimensions.

    From distributed shampoo.

    If there are some small dimensions, we collapse them:
    e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
         [1, 2, 768, 1, 2048] --> [2, 768, 2048]

    Args:
      shape_to_merge: Shape to merge small dimensions.
      max_dim: Maximal dimension of output shape used in merging.

    Returns:
      Merged shape.
    """
    if shape_to_merge and np.all(np.array(shape_to_merge) == 1):
        return [1]

    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape


def _sort_and_group_matrices(matrix_shapes: List[Tuple[int, ...]]):
    indexed_list = list(enumerate(matrix_shapes))
    sorted_indexed = sorted(indexed_list, key=lambda x: x[1])
    sorted_shapes = [shape for _, shape in sorted_indexed]
    change_indices = [original_index for original_index, _ in sorted_indexed]
    revert_indices = [0] * len(matrix_shapes)
    for new_pos, (original_index, _) in enumerate(sorted_indexed):
        revert_indices[original_index] = new_pos
    shape_groups = defaultdict(list)
    for i, shape in enumerate(sorted_shapes):
        shape_groups[shape].append(i)
    unique_sorted_shapes = list(shape_groups.keys())
    return unique_sorted_shapes, dict(shape_groups), change_indices, revert_indices


def _stack_matrices(array_list):
    in_tuple = isinstance(array_list, tuple)
    shapes = [arr.shape for arr in array_list]
    unique_shapes, shape_groups, change_indices, _ = _sort_and_group_matrices(shapes)
    sorted_arrays = [array_list[i] for i in change_indices]
    stacked_arrays = []
    for shape in unique_shapes:
        indices = shape_groups[shape]
        stacked = jnp.stack([sorted_arrays[i] for i in indices])
        stacked_arrays.append(stacked)
    if in_tuple:
        return tuple(stacked_arrays)
    return stacked_arrays


def _unstack_matrices(stacked_arrays, revert_indices):
    in_tuple = isinstance(stacked_arrays, tuple)
    unstacked = []
    for arr in stacked_arrays:
        unstacked.extend(jnp.split(arr, arr.shape[0]))
    array_list = [jnp.squeeze(unstacked[i], axis=0) for i in revert_indices]
    if in_tuple:
        return tuple(array_list)
    return array_list


if __name__ == "__main__":

    def _test_kron_optimizer():
        import optax

        # Test cases with different parameter shapes and configurations
        test_cases = [
            {
                "name": "Basic configuration",
                "params": {"w": jnp.ones((256, 256)), "b": jnp.ones(256)},
                "config": {"learning_rate": 0.001, "b1": 0.9},
            },
            {
                "name": "Weight decay and dtype configuration",
                "params": {"w": jnp.ones((128, 256)), "b": jnp.ones(128)},
                "config": {
                    "learning_rate": lambda n: 0.001
                    * 0.99**n,  # Test callable learning rate
                    "b1": 0.9,
                    "weight_decay": 0.01,
                    "weight_decay_mask": {"w": True, "b": False},
                    "mu_dtype": jnp.float32,
                    "precond_dtype": jnp.float32,
                },
            },
            {
                "name": "Memory modes and precision configuration",
                "params": {"w": jnp.ones((128, 128, 3, 64)), "b": jnp.ones(64)},
                "config": {
                    "learning_rate": 0.001,
                    "b1": 0.9,
                    "memory_save_mode": "one_diag",
                    "precond_update_precision": "bfloat16",
                    "precond_grads_precision": "float32",
                    "max_size_triangular": 4096,
                    "min_ndim_triangular": 3,
                },
            },
            {
                "name": "Scanned layers with custom update probability",
                "params": {
                    "layer1": {"w": jnp.ones((8, 256, 512)), "b": jnp.ones((8, 256))},
                    "layer2": {"w": jnp.ones((256, 128)), "b": jnp.ones(128)},
                },
                "config": {
                    "learning_rate": 0.001,
                    "b1": 0.9,
                    "preconditioner_update_probability": 0.5,  # Test constant probability
                    "scanned_layers": {
                        "layer1": {"w": True, "b": True},
                        "layer2": {"w": False, "b": False},
                    },
                    "lax_map_scanned_layers": True,
                    "lax_map_batch_size": 4,
                },
            },
            {
                "name": "Memory optimization features",
                "params": {
                    "w1": jnp.ones((256, 32, 32, 64)),
                    "w2": jnp.ones((512, 256)),
                },
                "config": {
                    "learning_rate": 0.001,
                    "b1": 0.9,
                    "merge_small_dims": True,
                    "max_merged_dim_size": 2048,
                    "partition_grads_into_blocks": True,
                    "block_size": 64,
                },
            },
        ]

        for test_case in test_cases:
            print(f"\nTesting {test_case['name']}...")
            print("Configuration:", test_case["config"])
            print("Parameters:", jax.tree.map(lambda x: x.shape, test_case["params"]))

            # Initialize optimizer
            optimizer = kron(**test_case["config"])
            opt_state = optimizer.init(test_case["params"])
            print("Optimizer state:", jax.tree.map(lambda x: x.shape, opt_state))

            # Create dummy gradients
            grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, test_case["params"])

            try:
                # Test single update
                updates, new_state = optimizer.update(
                    grads, opt_state, test_case["params"]
                )

                # Verify shapes match
                jax.tree.map(
                    lambda u, p: assert_shape_match(u.shape, p.shape),
                    updates,
                    test_case["params"],
                )

                # Apply updates
                new_params = optax.apply_updates(test_case["params"], updates)
                print("✓ Single update test passed")

                # Test multiple updates
                for i in range(3):
                    updates, new_state = optimizer.update(grads, new_state, new_params)
                    new_params = optax.apply_updates(new_params, updates)
                print("✓ Multiple updates test passed")

            except Exception as e:
                print(f"✗ Test failed: {str(e)}")
                raise e

    def assert_shape_match(shape1, shape2):
        """Helper function to verify shapes match"""
        assert shape1 == shape2, f"Shape mismatch: {shape1} vs {shape2}"

    _test_kron_optimizer()
