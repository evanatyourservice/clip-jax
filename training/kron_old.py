from typing import Any, List, Optional, Union, Callable
from functools import partial
import string
import numpy as np

import chex
import jax
from jax import vmap
import jax.numpy as jnp
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
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        b1: float, momentum parameter.
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
        scanned_layers: optional base.Params, tree of bool same structure as params
            indicating scanned layers. PSGD will vmap over the first dim.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge dimensions of tensors with
            more than 2 dimensions to improve compile times and preconditioner
            efficacy.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    preconditioner_lr = 0.1
    preconditioner_init_scale = 1.0

    def map_fn(do_map, fn, *args):
        """Maybe map a fn along first axis."""
        if do_map:
            if lax_map_scanned_layers:
                return jax.lax.map(
                    lambda xs: fn(*xs),
                    xs=args,
                    batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None,
                )
            else:
                return vmap(fn)(*args)
        else:
            return fn(*args)

    def init_fn(params):
        params = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda v: isinstance(v, (chex.Array, nn.Partitioned)),
        )

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)

        # momentum
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)

        # merge dimensions
        params, params_struct = jax.tree.flatten(params)
        scanned_layers_ = params_struct.flatten_up_to(scanned_layers_)
        if merge_small_dims:
            reshapers = [
                _merge_dims(p[0] if s else p) for p, s in zip(params, scanned_layers_)
            ]
            params = [
                map_fn(s, r[0], p)
                for s, r, p in zip(scanned_layers_, reshapers, params)
            ]

        # preconditioners
        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
            )[0]
            for t, s in zip(params, scanned_layers_)
        ]
        # broadcast for scanned layers
        Qs = [
            (
                jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), q
                )
                if s
                else q
            )
            for q, t, s in zip(Qs, params, scanned_layers_)
        ]
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
        
        def norm_grads(x):
            norm = jnp.linalg.norm(x)
            norm = jnp.where(norm == 0, 1, norm)
            return jnp.tanh(x / norm / 3.0) * 3.0

        updates = jax.tree.map(norm_grads, updates)

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

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
            reshapers = [
                _merge_dims(p[0] if s else p)
                for p, s in zip(momentum_updates, scanned_layers_)
            ]
            updates = [
                map_fn(s, r[0], p)
                for s, r, p in zip(scanned_layers_, reshapers, updates)
            ]
            momentum_updates = [
                map_fn(s, r[0], p)
                for s, r, p in zip(scanned_layers_, reshapers, momentum_updates)
            ]

        # get einsum expressions
        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(momentum_updates, scanned_layers_, Qs)
        ]

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precond_update_precision):
                # create random vectors
                key, subkey = jax.random.split(key)
                Vs_keys = jax.random.split(subkey, len(momentum_updates))
                Vs = [
                    jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                    for k, g in zip(Vs_keys, momentum_updates)
                ]

                # balance preconditioners about every 100 updates
                def balance_Qs(Qs: List[List[jax.Array]]):
                    def _balance_Q(Q: List[jax.Array]):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.prod(norms) ** (1 / len(norms))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return [
                        map_fn(s, _balance_Q, Q) if len(Q) > 1 else Q
                        for Q, s in zip(Qs, scanned_layers_)
                    ]

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) < 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)

                precond_update_in = momentum_updates

                # form conjB
                conjBs = [
                    map_fn(s, _conjB, Q, g, v)
                    for s, Q, g, v in zip(scanned_layers_, Qs, precond_update_in, Vs)
                ]

                # update Qs
                new_Qs = [
                    map_fn(
                        s,
                        partial(
                            _update_precond, exprs=exprs, precond_lr=preconditioner_lr
                        ),
                        Q,
                        g,
                        conjb,
                    )
                    for s, exprs, Q, g, conjb in zip(
                        scanned_layers_, expressions, Qs, precond_update_in, conjBs
                    )
                ]
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
                map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                for s, exprs, Q, g in zip(
                    scanned_layers_, expressions, Qs, momentum_updates
                )
            ]

        # trust region
        # precond_gs = jax.tree.map(trust_region, precond_gs)

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = [
                map_fn(s, r[1], p)
                for s, r, p in zip(scanned_layers_, reshapers, precond_gs)
            ]

        # box preconditioned grads
        if flax_partitioned:
            precond_gs = [
                u.replace_boxed(pg) for u, pg in zip(boxed_updates, precond_gs)
            ]

        # unflatten pytrees
        new_updates = grads_structure.unflatten(precond_gs)
        Qs = grads_structure.unflatten(Qs)

        # measure energy (x^2)
        energy = jax.tree_map(lambda x: jnp.mean(x**2), new_updates)
        energy = sum(jax.tree.leaves(energy)) / len(jax.tree.leaves(energy))
        jax.lax.cond(
            count_inc % 1000 == 0,
            lambda: jax.debug.print("Energy: {energy}", energy=energy),
            lambda: None,
        )

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
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        b1: float, momentum parameter.
        weight_decay: float, weight decay.
        weight_decay_mask: optional Any or callable, pytree of bool same structure
            as params with weight decay applied to True elements.
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
        scanned_layers: optional base.Params, tree of bool same structure as params
            indicating scanned layers. PSGD will vmap over the first dim.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge dimensions of tensors with
            more than 2 dimensions to improve compile times and preconditioner
            efficacy.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
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
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def _add_tiny(x):
    return x + jnp.finfo(x.dtype).tiny


def trust_region(x):
    x /= 1.5
    x = 0.1 * jnp.sign(x) * jnp.log(jnp.abs(x) + 1) + 0.9 * jnp.tanh(x)
    x *= 1.5
    return jnp.clip(x, -2, 2)


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
        exprA = ",->,"
        exprGs = [",->"]
        exprP = ",,->,"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

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

        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
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
    return [Q, (exprA, exprGs, exprP)]


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


def _merge_dims(arr: jax.Array) -> tuple:
    """Merges dimensions of tensor to improve efficiency.

    Adapted from original pytorch version (PSGD Affine).

    Args:
        arr: jax.Array, tensor to be reshaped.

    Returns:
        Tuple where first element is function that merges dimensions, second
        element is function that converts array back to original shape, and third
        element is the shape of x after merging dimensions.
    """

    def prod(arr):
        result = 1
        for a in arr:
            result *= a
        return result

    def permutations(p0):
        if len(p0) == 1:
            yield p0
        else:
            for i in range(len(p0)):
                for q in permutations(p0[:i] + p0[i + 1 :]):
                    yield p0[i], *q

    if arr.ndim <= 2:
        # these are fine, leave alone
        return lambda u: u, lambda v: v, arr.shape
    else:
        # higher order tensor, let's merge dimensions
        p0, s0 = tuple(range(arr.ndim)), arr.shape
        min_precond_size, opt_p, opt_s, opt_i = float("inf"), None, None, None
        for p in permutations(p0):
            s = tuple(s0[j] for j in p)
            for i in range(1, len(p)):
                if (new_size := prod(s[:i]) ** 2 + prod(s[i:]) ** 2) < min_precond_size:
                    min_precond_size = new_size
                    opt_p, opt_s, opt_i = p, s, i

        if opt_p == p0:  # no permutation is needed, just reshaping
            mtx_shape = (prod(s0[:opt_i]), prod(s0[opt_i:]))
            return (
                lambda u, shape=mtx_shape: u.reshape(shape),
                lambda v, shape=s0: v.reshape(shape),
                mtx_shape,
            )
        else:  # need both permutation and reshaping
            mtx_shape = (prod(opt_s[:opt_i]), prod(opt_s[opt_i:]))
            q = tuple(
                pair[1] for pair in sorted([(k, i) for (i, k) in enumerate(opt_p)])
            )
            return (
                lambda u, permute=opt_p, shape=mtx_shape: u.transpose(permute).reshape(
                    shape
                ),
                lambda v, permute=q, shape=opt_s: v.reshape(shape).transpose(permute),
                mtx_shape,
            )


def transform_gradients(x: jax.Array) -> jax.Array:
    if x.size <= 4:
        return x

    x2 = x * x
    kurt = jnp.mean(x2 * x2) / (jnp.mean(x2) ** 2)

    def transform(x):
        med = jnp.median(x)
        scale = jnp.median(jnp.abs(x - med)) * 1.4826 + 1e-12
        return med + scale * ((x - med) / scale) / (
            1.0 + jnp.abs((x - med) / (scale * 2.0))
        )

    # Replace hard threshold with smooth sigmoid transition
    alpha = jax.nn.sigmoid((kurt - 4.0) / 0.25)
    return alpha * transform(x) + (1 - alpha) * x


def compute_direction_change(x_before, x_after):
    """Calculate cosine similarity and angle between before/after vectors."""
    norm_before = jnp.sqrt(jnp.sum(x_before * x_before) + 1e-8)
    norm_after = jnp.sqrt(jnp.sum(x_after * x_after) + 1e-8)
    cos_sim = jnp.sum(x_before * x_after) / (norm_before * norm_after)
    angle_degrees = jnp.arccos(jnp.clip(cos_sim, -1.0, 1.0)) * 180 / jnp.pi
    return float(cos_sim), float(angle_degrees)


if __name__ == "__main__":
    def test_optimizer_and_distributions():
        """Test the optimizer with different momentum values and input distributions."""
        print("\nTesting Optimizer:")
        # Test parameters
        N = 64
        betas = [0.0, 0.9]
        steps = 4000
        
        # Add scale test distribution
        distributions = {
            "normal": lambda key: jax.random.normal(key, (N, N)),
            "uniform": lambda key: jax.random.uniform(key, (N, N), minval=-2, maxval=2),
            "laplace": lambda key: jax.random.laplace(key, (N, N)),
            "student_t": lambda key: jax.random.t(key, 3.0, (N, N)),
            "cauchy": lambda key: jax.random.cauchy(key, (N, N)),
        }

        for dist_name, dist_fn in distributions.items():
            print(f"\n=== Testing {dist_name.upper()} gradients ===")

            for beta in betas:
                print(f"\nMomentum beta={beta}")

                # Initialize state
                key = jax.random.PRNGKey(42)
                params = {"weight": jax.random.normal(key, (N, N))}
                opt = scale_by_kron(b1=beta)
                state = opt.init(params)

                @jax.jit
                def update_step(params, state, key):
                    key, subkey = jax.random.split(key)
                    grad = {"weight": dist_fn(subkey)}  # Use distribution-specific gradient
                    updates, new_state = opt.update(grad, state, params)
                    
                    # Print stats every 1000 steps
                    jax.lax.cond(
                        new_state["count"] % 1000 == 0,
                        lambda: jax.debug.print(
                            "Step: {step}, Energy: {energy:.4f}",
                            step=new_state["count"],
                            energy=jax.tree_map(lambda x: jnp.mean(x**2), updates)["weight"]
                        ),
                        lambda: None
                    )
                    
                    return updates, new_state, key

                # Run updates
                for _ in range(steps):
                    updates, state, key = update_step(params, state, key)
                    params = jax.tree.map(lambda p, u: p - u, params, updates)

                # Print final stats
                print(f"\nFinal Statistics:")
                print(f"Energy: {jax.tree_map(lambda x: jnp.mean(x**2), updates)['weight']:.4f}")
                
                # Print condition numbers
                Qs = state["Qs_preconditioners"]["weight"]
                for idx, Q in enumerate(Qs):
                    P = Q.T @ Q
                    s = jnp.linalg.svd(P, compute_uv=False)
                    condition_number = s[0] / s[-1]
                    print(f"Q{idx} condition number: {condition_number:.4f}")

        print("\nTesting Distribution Transforms:")
        # Generate test distributions
        key = jax.random.PRNGKey(42)
        N = 10000
        key, *subkeys = jax.random.split(key, 6)
        test_distributions = {
            "normal": jax.random.normal(subkeys[0], (N,)),
            "uniform": jax.random.uniform(subkeys[1], (N,), minval=-2, maxval=2),
            "laplace": jax.random.laplace(subkeys[2], (N,)),
            "student_t": jax.random.t(subkeys[3], 3.0, (N,)),
            "cauchy": jax.random.cauchy(subkeys[4], (N,)),
        }
        
        def compute_stats(x):
            """Calculate distribution statistics."""
            centered = x - jnp.mean(x)
            var = jnp.mean(centered**2)
            kurt = jnp.mean(centered**4) / (var**2)
            p99, p50 = jnp.percentile(jnp.abs(x), jnp.array([99, 50]))
            return {
                "mean": float(jnp.mean(x)),
                "std": float(jnp.std(x)),
                "range": [float(jnp.min(x)), float(jnp.max(x))],
                "kurtosis": float(kurt),
                "tail_ratio": float(p99 / (p50 + 1e-8))
            }
        
        # Test each distribution
        for name, data in test_distributions.items():
            transformed = transform_gradients(data)
            before = compute_stats(data)
            after = compute_stats(transformed)
            
            print(f"\n{name.upper()} Distribution:")
            print("Before transformation:")
            print(f"  Mean: {before['mean']:.3f}")
            print(f"  Std: {before['std']:.3f}")
            print(f"  Range: [{before['range'][0]:.3f}, {before['range'][1]:.3f}]")
            print(f"  Kurtosis: {before['kurtosis']:.3f}")
            print(f"  Tail ratio (p99/p50): {before['tail_ratio']:.3f}")
            
            print("After transformation:")
            print(f"  Mean: {after['mean']:.3f}")
            print(f"  Std: {after['std']:.3f}")
            print(f"  Range: [{after['range'][0]:.3f}, {after['range'][1]:.3f}]")
            print(f"  Kurtosis: {after['kurtosis']:.3f}")
            print(f"  Tail ratio (p99/p50): {after['tail_ratio']:.3f}")

        print("\nTesting Direction Preservation:")
        for name, data in test_distributions.items():
            transformed = transform_gradients(data)
            cos_sim, angle = compute_direction_change(data, transformed)
            print(f"\n{name.upper()} Distribution:")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  Angle change: {angle:.2f}°")

    test_optimizer_and_distributions()
