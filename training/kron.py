import os
from typing import Any, List, Optional, Union, Callable, Tuple
from functools import partial
import string
from collections import defaultdict
import numpy as np

import chex
import jax
from jax import vmap
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint
import flax.linen as nn
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.clip(
            max_prob * jnp.exp(-decay * (n - flat_start)), min_prob, max_prob
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
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec[str, str]] = None,
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
        target_merged_dim_size: int, target product of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for memory efficiency.
        block_size: int, size of partitions to use for memory efficiency.
        params_sharding: pytree same structure as params of PartitionSpec.
        preconditioner_sharding: partition spec for preconditioner matrices
            PartitionSpec(str | None, str | None).

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
        have_params_sharding = params_sharding is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        params = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda v: isinstance(
                v, (chex.Array, nn.Partitioned, jax.ShapeDtypeStruct)
            ),
        )

        # params sharding
        params_sharding_ = params_sharding
        if params_sharding is None:
            params_sharding_ = jax.tree.map(lambda _: None, params)
        if have_params_sharding:
            # extend partition specs to all dims
            params_sharding_ = jax.tree.map(
                lambda p, sh: (
                    PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh))))
                    if sh is not None
                    else None
                ),
                params,
                params_sharding_,
            )

        # scanned layers
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
            # apply params sharding to momentum buffer
            if have_params_sharding:
                mu = safe_sharding_constraint(mu, params_sharding_)

        params, params_struct = jax.tree.flatten(params)
        scanned_layers_ = params_struct.flatten_up_to(scanned_layers_)
        scanned_sizes = params_struct.flatten_up_to(scanned_sizes)
        params_without_scan = [
            _first_n_dims(p, int(s)) for p, s in zip(params, scanned_layers_)
        ]
        dim_diag = [
            _get_preconditioner_types(
                p.shape, max_size_triangular, min_ndim_triangular, memory_save_mode
            )
            for p in params_without_scan
        ]  # tuple of lists of booleans, one bool per dim indicating diag preconds
        params_sharding_ = params_struct.flatten_up_to(params_sharding_)
        scanned_dim_sharding = [
            PartitionSpec(sh[0]) if (s and sh is not None) else None
            for s, sh in zip(scanned_layers_, params_sharding_)
        ]
        sharding_without_scan = [
            PartitionSpec(*(sh[1:] if s else sh)) if sh is not None else None
            for s, sh in zip(scanned_layers_, params_sharding_)
        ]

        # merge dimensions
        if merge_small_dims:
            output = [
                _merge_small_dims(p.shape, target_merged_dim_size, dd, sh)
                for p, dd, sh in zip(
                    params_without_scan, dim_diag, sharding_without_scan
                )
            ]
            merged_shapes, dim_diag, sharding_without_scan = map(list, zip(*output))
            params_without_scan = [
                jnp.reshape(p, ns) for p, ns in zip(params_without_scan, merged_shapes)
            ]

        # partition grads into blocks
        if partition_grads_into_blocks:
            partitioners = [
                BlockPartitioner(p.shape, block_size, dd)
                for p, dd in zip(params_without_scan, dim_diag)
            ]
            # this becomes list of tuples where each tuple contains layer's partitions
            params_without_scan = [
                p_cls.partition(p)
                for p_cls, p in zip(partitioners, params_without_scan)
            ]
            # pad and stack tuples of partitions into single array
            params_without_scan = [
                _pad_and_stack_matrices(p, block_size) for p in params_without_scan
            ]

        # initialize preconditioners
        output = [
            _init_Q_exprs(
                _first_n_dims(p, int(partition_grads_into_blocks)),
                preconditioner_init_scale,
                dd,
                precond_dtype,
                precond_sharding=preconditioner_sharding,
                param_sharding=sh,
            )
            for p, dd, sh in zip(params_without_scan, dim_diag, sharding_without_scan)
        ]
        Qs = [x[0] for x in output]
        Qs_sharding = [x[2] for x in output]
        if have_qs_sharding:
            # add scan and stack dims to sharding
            Qs_sharding = [
                (
                    [
                        (
                            PartitionSpec(
                                *(
                                    sds + (None,) + qs
                                    if sds is not None
                                    else (None,) + qs
                                )
                            )
                            if qs is not None
                            else None
                        )
                        for qs in qss
                    ]
                    if qss is not None
                    else None
                )
                for sds, qss in zip(scanned_dim_sharding, Qs_sharding)
            ]
        # broadcast for stacks and scans
        all_Qs = []
        for q, p, s in zip(Qs, params_without_scan, scanned_sizes):
            new_q = q
            if partition_grads_into_blocks:
                # add leading dim for stacked partitions
                new_q = jax.tree.map(
                    lambda x: jnp.repeat(jnp.expand_dims(x, 0), p.shape[0], axis=0),
                    new_q,
                )
            if s > 0:
                # add leading dim if we're scanning this layer
                new_q = jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), new_q
                )
            all_Qs.append(new_q)
        # for a stacked and scanned layer, it now has two leading dims to be vmapped
        Qs = all_Qs
        if have_qs_sharding:
            Qs = safe_sharding_constraint(Qs, Qs_sharding)
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
        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = jax.random.fold_in(jax.random.PRNGKey(42), state["count"])

        have_params_sharding = params_sharding is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        boxed_updates, grads_structure = jax.tree.flatten(
            updates,
            is_leaf=lambda v: isinstance(
                v, (chex.Array, nn.Partitioned, jax.ShapeDtypeStruct)
            ),
        )
        flax_partitioned = False
        if isinstance(boxed_updates[0], nn.Partitioned):
            flax_partitioned = True
            updates = [u.unbox() for u in boxed_updates]
            updates = grads_structure.unflatten(updates)

        # params sharding
        params_sharding_ = params_sharding
        if params_sharding is None:
            params_sharding_ = jax.tree.map(lambda _: None, updates)
        if have_params_sharding:
            # extend partition specs to all dims
            params_sharding_ = jax.tree.map(
                lambda p, sh: (
                    PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh))))
                    if sh is not None
                    else None
                ),
                updates,
                params_sharding_,
            )

        # scanned layers
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        # update probability can be scheduled
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
            if have_params_sharding:
                mu = safe_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        momentum_updates = grads_structure.flatten_up_to(momentum_updates)
        Qs = grads_structure.flatten_up_to(state["Qs_preconditioners"])
        scanned_layers_ = grads_structure.flatten_up_to(scanned_layers_)
        dim_diag = [
            _get_preconditioner_types(
                p.shape[1:] if s else p.shape,
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
            )
            for p, s in zip(updates, scanned_layers_)
        ]  # tuple of lists of booleans, one bool per dim indicating diag preconds
        params_sharding_ = grads_structure.flatten_up_to(params_sharding_)
        scanned_dim_sharding = [
            PartitionSpec(sh[0]) if (s and sh is not None) else None
            for s, sh in zip(scanned_layers_, params_sharding_)
        ]
        sharding_without_scan = [
            PartitionSpec(*(sh[1:] if s else sh)) if sh is not None else None
            for s, sh in zip(scanned_layers_, params_sharding_)
        ]

        # merge dimensions
        merged_params_sharding = params_sharding_
        if merge_small_dims:
            original_shapes = [
                p.shape[1:] if s else p.shape
                for p, s in zip(momentum_updates, scanned_layers_)
            ]
            output = [
                _merge_small_dims(os, target_merged_dim_size, dd, sh)
                for os, dd, sh in zip(original_shapes, dim_diag, sharding_without_scan)
            ]
            merged_shapes, dim_diag, sharding_without_scan = map(list, zip(*output))
            momentum_updates = [
                _map_fn(False, 0, int(s), lambda p, shape=ns: jnp.reshape(p, shape), p)
                for s, p, ns in zip(scanned_layers_, momentum_updates, merged_shapes)
            ]
            if have_params_sharding:
                # scanned dim sharding + new merged sharding
                merged_params_sharding = [
                    (
                        PartitionSpec(*(sds + sws if sds is not None else sws))
                        if sws is not None
                        else None
                    )
                    for sds, sws in zip(scanned_dim_sharding, sharding_without_scan)
                ]
                # constrain sharding
                momentum_updates = safe_sharding_constraint(
                    momentum_updates, merged_params_sharding
                )

        # partition grads into blocks
        partitioned_sharding = merged_params_sharding
        n_dims_to_map = [int(s) for s in scanned_layers_]
        if partition_grads_into_blocks:
            partitioners = [
                BlockPartitioner(p.shape[1:] if s else p.shape, block_size, dd)
                for s, p, dd in zip(scanned_layers_, momentum_updates, dim_diag)
            ]
            # this becomes list of tuples where each tuple contains layer's partitions
            momentum_updates = [
                _map_fn(False, 0, int(s), p_cls.partition, p)
                for s, p, p_cls in zip(scanned_layers_, momentum_updates, partitioners)
            ]
            partitioned_shapes = [
                jax.tree.map(lambda x: x.shape[1:] if s else x.shape, p)
                for s, p in zip(scanned_layers_, momentum_updates)
            ]
            if have_params_sharding:
                # constrain partitions to same sharding as entire layer
                momentum_updates = [
                    jax.tree.map(lambda x: safe_sharding_constraint(x, mps), p)
                    for p, mps in zip(momentum_updates, merged_params_sharding)
                ]
            # pad and stack tuples of partitions into single array
            momentum_updates = [
                _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    p,
                )
                for s, p in zip(scanned_layers_, momentum_updates)
            ]
            if have_params_sharding:
                # add dim to sharding for new stacked dim
                partitioned_sharding = [
                    (
                        PartitionSpec(
                            *(mps[:1] + (None,) + mps[1:] if s else (None,) + mps)
                        )
                        if mps
                        else PartitionSpec(None)
                    )
                    for s, mps in zip(scanned_layers_, merged_params_sharding)
                ]
                # constrain sharding
                momentum_updates = safe_sharding_constraint(
                    momentum_updates, partitioned_sharding
                )
            n_dims_to_map = [1 + int(s) for s in scanned_layers_]

        # get einsum expressions and Qs sharding
        Qs_sharding = [None] * len(momentum_updates)
        exprs_and_sharding = [
            _init_Q_exprs(
                _first_n_dims(g, nm),
                preconditioner_init_scale,
                dd,
                precond_dtype,
                existing_Q=jax.tree.map(lambda q: _first_n_dims(q, nm), q),
                precond_sharding=preconditioner_sharding,
                param_sharding=sh,
            )
            for g, q, dd, sh, nm in zip(
                momentum_updates, Qs, dim_diag, sharding_without_scan, n_dims_to_map
            )
        ]  # list of lists
        expressions = [x[0] for x in exprs_and_sharding]
        if have_qs_sharding:
            Qs_sharding = [x[1] for x in exprs_and_sharding]
            # add scan and stack dims to sharding
            Qs_sharding = [
                (
                    [
                        (
                            PartitionSpec(
                                *(
                                    sds + (None,) + qs
                                    if sds is not None
                                    else (None,) + qs
                                )
                            )
                            if qs is not None
                            else None
                        )
                        for qs in qss
                    ]
                    if qss is not None
                    else None
                )
                for sds, qss in zip(scanned_dim_sharding, Qs_sharding)
            ]

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precond_update_precision):
                # create random vectors
                key, subkey = jax.random.split(key)
                Vs = _tree_random_like(subkey, momentum_updates)
                # apply params sharding to random vectors
                if have_params_sharding:
                    Vs = safe_sharding_constraint(Vs, partitioned_sharding)

                # balance preconditioners about every 100 updates
                def balance_Qs(Qs_to_bal):
                    def _balance_Q(Q):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.exp(jnp.mean(jnp.log(norms)))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return [
                        _map_fn(False, 0, nm, _balance_Q, Q) if len(Q) > 1 else Q
                        for Q, nm in zip(Qs_to_bal, n_dims_to_map)
                    ]

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) <= 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)
                if have_qs_sharding:
                    Qs = safe_sharding_constraint(Qs, Qs_sharding)

                # form conjB
                conjBs = [
                    _map_fn(lax_map, bs, nm, _conjB, Q, g, v)
                    for g, Q, v, nm in zip(momentum_updates, Qs, Vs, n_dims_to_map)
                ]
                # if have_params_sharding:
                #     conjBs = safe_sharding_constraint(conjBs, partitioned_sharding)

                # update Qs and constrain sharding
                new_Qs = [
                    _map_fn(
                        lax_map,
                        bs,
                        nm,
                        partial(
                            _update_precond, exprs=expr_sh, precond_lr=preconditioner_lr
                        ),
                        Q,
                        g,
                        conjb,
                    )
                    for expr_sh, g, Q, conjb, nm in zip(
                        expressions, momentum_updates, Qs, conjBs, n_dims_to_map
                    )
                ]
                return new_Qs

        # update preconditioner deterministically
        update_counter_inc = safe_int32_increment(state["update_counter"])
        do_update = update_counter_inc >= 1 / update_prob_in
        update_counter_inc = jnp.where(do_update, 0, update_counter_inc)
        key, subkey = jax.random.split(key)
        new_Qs = jax.lax.cond(
            do_update, update_preconditioner, lambda _, qs: qs, subkey, Qs
        )
        if have_qs_sharding:
            new_Qs = safe_sharding_constraint(new_Qs, Qs_sharding)

        # precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            # precondition with stale Qs
            precond_gs = [
                _map_fn(lax_map, bs, nm, partial(_precond_grad, exprs=expr_sh), Q, g)
                for expr_sh, g, Q, nm in zip(
                    expressions, momentum_updates, Qs, n_dims_to_map
                )
            ]
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, partitioned_sharding)

        # unstack and merge partitioned grads
        if partition_grads_into_blocks:
            precond_gs = [
                _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=os: _unstack_and_unpad_matrices(p, shapes),
                    p,
                )
                for s, p, os in zip(scanned_layers_, precond_gs, partitioned_shapes)
            ]  # list of tuples
            if have_params_sharding:
                precond_gs = [
                    jax.tree.map(lambda x: safe_sharding_constraint(x, mps), p)
                    for p, mps in zip(precond_gs, merged_params_sharding)
                ]
            precond_gs = [
                _map_fn(False, 0, int(s), part.merge_partitions, p)
                for s, p, part in zip(scanned_layers_, precond_gs, partitioners)
            ]
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(
                    precond_gs, merged_params_sharding
                )

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = [
                _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), p)
                for s, os, p in zip(scanned_layers_, original_shapes, precond_gs)
            ]
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, params_sharding_)

        # box preconditioned grads
        if flax_partitioned:
            precond_gs = [
                u.replace_boxed(pg) for u, pg in zip(boxed_updates, precond_gs)
            ]

        # unflatten pytrees
        new_updates = grads_structure.unflatten(precond_gs)
        new_Qs = grads_structure.unflatten(new_Qs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        new_Qs = otu.tree_cast(new_Qs, precond_dtype)
        state = dict(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=new_Qs,
            update_counter=update_counter_inc,
        )

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
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec[str, str]] = None,
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
        target_merged_dim_size: int, target product of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for memory efficiency.
        block_size: int, size of partitions to use for memory efficiency.
        params_sharding: pytree same structure as params of PartitionSpec.
        preconditioner_sharding: partition spec for preconditioner matrices
            PartitionSpec(str | None, str | None).

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
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_size: int, min_ndim: int, mem_save_mode: Optional[str]
) -> List[bool]:
    if len(shape) == 0:
        return True

    if mem_save_mode is None:
        dim_diag = [False for _ in shape]
    elif mem_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        dim_diag[rev_sorted_dims[0]] = True
    elif mem_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid mem_save_mode: {mem_save_mode}, must be one of "
            "[None, 'one_diag', 'all_diag']"
        )

    for i, size in enumerate(shape):
        if size == 1 or size > max_size or len(shape) < min_ndim:
            dim_diag[i] = True

    return dim_diag


def _init_Q_exprs(
    t,
    scale,
    dim_diag,
    dtype,
    existing_Q=None,
    precond_sharding=None,
    param_sharding=None,
):
    """For a scalar or tensor `t`, we initialize its preconditioner `Q` and
    reusable contraction expressions for updating `Q` and preconditioning gradient.

    Args:
        t: Input tensor or scalar
        scale: Scale factor for initialization
        dim_diag: Diagonal flags for each dimension.
        dtype: Data type for preconditioners
        existing_Q: Optional existing preconditioners to reuse
        precond_sharding: Optional sharding spec for preconditioners
        param_sharding: Optional sharding spec for input tensor
    """
    have_qs_sharding = precond_sharding is not None or param_sharding is not None
    sharding_out = None
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

        if have_qs_sharding:
            sharding_out = [PartitionSpec()]
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )
        scale = scale ** (1 / len(shape))
        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

        params_specs = param_sharding
        if param_sharding is None:
            params_specs = PartitionSpec(*((None,) * len(shape)))
        if have_qs_sharding:
            sharding_out = [PartitionSpec(None)] * len(shape)

        for i, (size, dim_d, dim_sh) in enumerate(zip(shape, dim_diag, params_specs)):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    q = scale * jnp.ones(size, dtype=dtype)
                    Q.append(q)

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
                if have_qs_sharding:
                    # infer a so-so sharding scheme from params if nothing specified
                    q_sharding = (
                        precond_sharding
                        if precond_sharding is not None
                        else PartitionSpec(dim_sh)
                    )
                    sharding_out[i] = q_sharding
                if existing_Q is None:
                    q = scale * jnp.eye(size, dtype=dtype)
                    if have_qs_sharding:
                        q = safe_sharding_constraint(q, q_sharding)
                    Q.append(q)

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
        return (exprA, exprGs, exprP), sharding_out
    return Q, (exprA, exprGs, exprP), sharding_out


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


def safe_sharding_constraint(x, sharding):
    if sharding is None:
        return x
    else:
        return with_sharding_constraint(x, sharding)


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
    # adopted from optax
    tree_def = jax.tree.structure(target_tree)
    keys = jax.random.split(rng_key, tree_def.num_leaves)
    keys_tree = jax.tree.unflatten(tree_def, keys)
    return jax.tree.map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype), target_tree, keys_tree
    )


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size, dim_diag):
        assert len(dim_diag) == len(
            param_shape
        ), "dim_diag must have same length as param_shape"
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


def partitions(lst):
    """Generate all partitions of a list."""
    if not lst:
        yield [[]]
    else:
        for i in range(len(lst)):
            for part in partitions(lst[i + 1 :]):
                yield [lst[: i + 1]] + part


def _merge_small_dims(
    shape_to_merge, max_dim, dim_diag, sharding_to_merge=None
) -> Tuple[List[int], List[bool], Optional[Tuple]]:
    """Merge small dimensions and their corresponding sharding specs and diag flags.

    Args:
        shape_to_merge: Shape to merge small dimensions.
        max_dim: Target dimension size for merged dimensions.
        dim_diag: Diagonal flags for each dimension.
        sharding_to_merge: Optional partition spec to merge alongside shape.

    Returns:
        Tuple of (merged shape, merged diag flags, merged sharding).
    """
    if not shape_to_merge or np.all(np.array(shape_to_merge) == 1):
        return [1], [True], None

    def dim2loss(d, dim0=max_dim):
        """A heuristic map from dim to loss with the least loss occurs at dim0."""
        loss = 0
        if d < dim0:
            loss += np.log2(dim0 / d)
            too_small = dim0 / 8
            if d < too_small:
                loss += 100 * np.log2(too_small / d)
        else:
            loss += 10 * np.log2(d / dim0)
            too_large = 8 * dim0
            if d > too_large:
                loss += 1000 * np.log2(d / too_large)
        return loss

    best_loss = float("inf")
    best_partition = None

    for p in partitions(list(range(len(shape_to_merge)))):
        loss = 0
        merged = []
        for group in p:
            if not group:
                continue
            d = np.prod([shape_to_merge[i] for i in group])
            loss += dim2loss(d)
            merged.append(group)

        if loss < best_loss:
            best_loss = loss
            best_partition = merged

    merged_shape = []
    merged_diag = []
    merged_sharding = []

    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))
        merged_diag.append(all(dim_diag[i] for i in group))
        if sharding_to_merge:
            group_shardings = [sharding_to_merge[i] for i in group]
            valid_shardings = [s for s in group_shardings if s is not None]

            if len(valid_shardings) > 1:
                merged_sharding.append(tuple(valid_shardings))
            elif len(valid_shardings) == 1:
                merged_sharding.append(valid_shardings[0])
            else:
                merged_sharding.append(None)

    return (
        merged_shape,
        merged_diag,
        PartitionSpec(*merged_sharding) if sharding_to_merge else None,
    )


def _pad_and_stack_matrices(array_list, block_size):
    shapes = [arr.shape for arr in array_list]
    max_dims = [max(shape[i] for shape in shapes) for i in range(len(shapes[0]))]
    padded_shape = [-(-dim // block_size) * block_size for dim in max_dims]
    padded_arrays = []
    for arr in array_list:
        pad_width = [(0, padded_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        padded = jnp.pad(arr, pad_width, mode="constant", constant_values=0)
        padded_arrays.append(padded)
    return jnp.stack(padded_arrays)


def _unstack_and_unpad_matrices(stacked_array, original_shapes):
    unstacked = jnp.split(stacked_array, stacked_array.shape[0], axis=0)
    unpadded = []
    for arr, orig_shape in zip(unstacked, original_shapes):
        slices = tuple(slice(0, dim) for dim in orig_shape)
        unpadded.append(jnp.squeeze(arr, axis=0)[slices])
    return tuple(unpadded)


# unused fns:
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


def profile_kron():
    devices = jax.devices()
    mesh_shape = (len(devices) // 2, 2)
    device_mesh = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, ('data', 'model'))
    
    rng = jax.random.PRNGKey(0)
    hidden_dim = 4096
    num_layers = 24
    
    def create_fake_params():
        params = []
        for _ in range(num_layers):
            layer_params = {
                'attention': {
                    'qkv': jax.random.normal(rng, (hidden_dim, 3 * hidden_dim)),
                    'output': jax.random.normal(rng, (hidden_dim, hidden_dim)),
                },
                'mlp': {
                    'fc1': jax.random.normal(rng, (hidden_dim, 4 * hidden_dim)),
                    'fc2': jax.random.normal(rng, (4 * hidden_dim, hidden_dim)),
                }
            }
            params.append(layer_params)
        return params
    
    def create_sharding_specs():
        layer_spec = {
            'attention': {
                'qkv': PartitionSpec('model', 'data'),
                'output': PartitionSpec('data', 'model'),
            },
            'mlp': {
                'fc1': PartitionSpec('model', 'data'),
                'fc2': PartitionSpec('data', 'model'),
            }
        }
        return [layer_spec for _ in range(num_layers)]
    
    def create_scan_specs():
        layer_spec = {
            'attention': {
                'qkv': True,
                'output': True,
            },
            'mlp': {
                'fc1': True,
                'fc2': True,
            }
        }
        return [layer_spec for _ in range(num_layers)]
    
    params = create_fake_params()
    params_sharding = create_sharding_specs()
    scanned_layers = create_scan_specs()
    
    optimizer = kron(
        learning_rate=0.001,
        b1=0.9,
        normalize_grads=True,
        preconditioner_update_probability=1.0,
        memory_save_mode='one_diag',
        merge_small_dims=True,
        target_merged_dim_size=2048,
        partition_grads_into_blocks=True,
        block_size=512,
        params_sharding=params_sharding,
        preconditioner_sharding=PartitionSpec('data', 'model'),
        scanned_layers=scanned_layers,
        lax_map_scanned_layers=True,
        lax_map_batch_size=8,
    )
    
    with mesh:
        opt_state = optimizer.init(params)
    
    grads = jax.tree.map(lambda x: jax.random.normal(rng, x.shape, x.dtype), params)
    
    @jax.jit
    def opt_step(opt_state, grads):
        updates, new_state = optimizer.update(grads, opt_state, params)
        return new_state
    
    with mesh:
        state = opt_step(opt_state, grads)
    
    for step in range(20):
        if step == 5:
            if jax.process_index() == 0:
                jax.profiler.start_trace("/tmp/kron_profile")
        
        with jax.profiler.StepTraceAnnotation("optimizer_step", step_num=step):
            with mesh:
                state = opt_step(state, grads)
                
        if step == 10:
            if jax.process_index() == 0:
                jax.profiler.stop_trace()

if __name__ == "__main__":
    profile_kron()
