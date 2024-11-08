import logging
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pprint import pformat
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import multihost_utils
from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.pjit import pjit
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, PartitionSpec
from kron_old import precond_update_prob_schedule, scale_by_kron
from precondition_local.distributed_shampoo import GraftingType, distributed_shampoo
from tqdm import tqdm
from transformers import HfArgumentParser

LogicalAxisRules = Sequence[Tuple[str, Optional[str]]]

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    pass



@dataclass
class ModelArguments:
    pass

@dataclass
class TrainingArguments:
    no_cache: bool = field(default=False, metadata={"help": "Uses jax cache."})
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    optim: str = field(
        default="kron",
        metadata={"help": ('The optimizer to use. Can be "distributed_shampoo" (default), "adam" or "kron"')},
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay applied to parameters."})
    beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for Adam & Distributed Shampoo."},
    )
    beta2: float = field(
        default=0.99,
        metadata={"help": "Beta2 for for Adam & Distributed Shampoo."},
    )
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    block_size_encoder: int = field(
        default=1024,
        metadata={"help": "Chunked size for large encoder layers with Distributed Shampoo."},
    )
    block_size_unet: int = field(
        default=768,
        metadata={"help": "Chunked size for large unet layers with Distributed Shampoo."},
    )
    caspr_variant: bool = field(
        default=False,
        metadata={"help": "Use CASPR variant of Distributed Shampoo."},
    )
    preconditioning_compute_steps: int = field(
        default=20, metadata={"help": "Number of steps to update preconditioner."}
    )
    skip_preconditioning_dim_size_gt: int = field(
        default=4096,
        metadata={"help": "Max size for preconditioning with Distributed Shampoo."},
    )
    graft_type: str = field(
        default="rmsprop_normalized",
        metadata={
            "help": (
                "The type of grafting to use. Can be 'rmsprop_normalized' (default),"
                " 'rmsprop', 'adagrad', 'adagrad_normalized', 'sgd' or 'sqrt_n'"
            )
        },
    )
    clip_by_scaled_gradient_norm: float = field(
        default=False,
        metadata={"help": "Clip by scaled gradient norm, only used with shampoo and rmsprop grafting."},
    )
    kron_mem_save_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Kron memory save mode, can be None, 'one_diag', or 'all_diag'."},
    )
    kron_merge_small_dims: bool = field(
        default=True,
        metadata={"help": "Merge small dimensions for Kron."},
    )
    nesterov: bool = field(
        default=False,
        metadata={"help": "Use Nesterov momentum for Distributed Shampoo."},
    )
    exponent_override: int = field(
        default=0, metadata={"help": "Override the exponent used in matrix inverse for Distributed Shampoo."}
    )
    eigh: bool = field(
        default=False,
        metadata={"help": ("Uses eigen decomposition for inverse-pth root Shampoo).")},
    )
    optim_quantized: bool = field(
        default=False,
        metadata={"help": ("Whether to quantize optimizer (only supported with Distributed Shampoo).")},
    )
    shard_shampoo_across: str = field(
        default="2d",
        metadata={"help": ("Whether to shard the optimizer across data devices, model devices or both (2d).")},
    )
    num_train_epochs: int = field(default=10, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    lr_decay: str = field(
        default=None,
        metadata={
            "help": (
                "Decay to be used in the learning rate scheduler. Can be None (default), linear, cosine or exponential."
            )
        },
    )
    compact_params: int = field(
        default=0,
        metadata={
            "help": ("Whether to compact parameters for optimizer, and the maximum number of parameters per tensor.")
        },
    )
    lr_transition_steps: int = field(
        default=None,
        metadata={"help": ("Number of transition steps associated with learning rate decay when using decay.")},
    )
    lr_decay_rate: float = field(
        default=None,
        metadata={"help": ("Decay rate associated with learning rate when using exponential decay.")},
    )
    lr_staircase: bool = field(
        default=False,
        metadata={"help": ("Whether to use staircase or continuous learning rate when using exponential decay.")},
    )
    lr_offset: int = field(
        default=0,
        metadata={"help": "Number of steps to offset learning rate and keep it at 0."},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=100, metadata={"help": "Run an evaluation every X steps."})
    debug: bool = field(
        default=False,
        metadata={"help": "Output more variables for debugging."},
    )
    log_norm: bool = field(
        default=True,
        metadata={"help": "Log parameters and gradients norm."},
    )
    log_histogram: int = field(
        default=False,
        metadata={"help": ("Log parameters and gradients histograms. Slows down training.")},
    )
    trainable_params_mode: str = field(
        default=None,
        metadata={"help": ("Custom trainable parameters mode.")},
    )

    seed_model: int = field(
        default=42,
        metadata={"help": ("Random seed for the model that will be set at the beginning of" " training.")},
    )

    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The wandb entity to use (for teams)."},
    )
    wandb_project: str = field(
        default="unet",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_job_type: str = field(
        default="train",
        metadata={"help": "The name of the wandb job type."},
    )

    assert_TPU_available: bool = field(
        default=False,
        metadata={"help": "Verify that TPU is not in use."},
    )

    mp_devices: int = field(
        default=1,
        metadata={
            "help": (
                "Number of devices required for model parallelism. The other dimension"
                " of available devices is used for data parallelism."
            )
        },
    )
    activation_partitioning_dims: int = field(
        default=1,
        metadata={"help": ("Number of dimensions for activation partitioning (1 or 2).")},
    )
    parameter_partitioning_dims: int = field(
        default=1,
        metadata={"help": ("Number of dimensions for activation partitioning (1 or 2).")},
    )

    skip_update: bool = field(
        default=False,
        metadata={"help": "Skip update if parameters are non finite."},
    )

    do_profile: bool = field(
        default=False,
        metadata={"help": "Profile performance of training loop."},
    )
    do_lower: bool = field(
        default=False,
        metadata={"help": "Profile performance of training loop."},
    )
    do_test_steps: int = field(
        default=False,
        metadata={"help": "Run script for only a few steps."},
    )

    dp_devices: int = field(init=False)
    log_norm_steps: int = field(init=False)

    def __post_init__(self):
        if self.assert_TPU_available:
            assert jax.local_device_count() > 1, "TPUs in use, please check running processes"
        assert self.lr_decay in [
            None,
            "linear",
            "cosine",
            "exponential",
        ], f"Selected learning rate decay not supported: {self.lr_decay}"
        if self.log_norm is True:
            self.log_norm_steps = self.logging_steps
        else:
            self.log_norm_steps = False
        if self.log_histogram is True:
            self.log_histogram_steps = self.logging_steps
        else:
            self.log_histogram_steps = False
        if not self.do_train:
            # eval only
            self.num_train_epochs = 1
        if self.do_profile:
            self.start_profile = 3
            self.end_profile = 5
            self.do_test_steps = self.end_profile + 2
        if self.compact_params:
            assert self.optim in ["kron"], "Compact params only supported with Kron optimizer."
        assert self.optim in [
            "distributed_shampoo",
            "adam",
            "kron",
        ], f"Unknown optimizer {self.optim}"
        assert self.graft_type in [
            "rmsprop_normalized",
            "rmsprop",
            "adagrad",
            "adagrad_normalized",
            "sgd",
            "sqrt_n",
        ], f"Selected graft type not supported: {self.graft_type}"
        assert self.shard_shampoo_across in [
            "data",
            "model",
            "2d",
        ], f"Shard shampoo across {self.shard_shampoo_across} not supported."
        assert self.mp_devices > 0, "Number of devices for model parallelism must be > 0"
        assert jax.device_count() % self.mp_devices == 0, (
            f"Number of available devices ({jax.device_count()} must be divisible by"
            f" number of devices used for model parallelism ({self.mp_devices})."
        )
        assert self.parameter_partitioning_dims in [1, 2], (
            f"Number of dimensions for parameter partitioning must be 1 or 2"
            f" (got {self.parameter_partitioning_dims})."
        )
        assert self.activation_partitioning_dims in [1, 2], (
            f"Number of dimensions for activation partitioning must be 1 or 2"
            f" (got {self.activation_partitioning_dims})."
        )
        self.dp_devices = jax.device_count() // self.mp_devices



def flat_args(model_args, data_args, training_args):
    """Flatten arguments to be able to use config files"""
    args = asdict(model_args)
    args.update(asdict(data_args))
    args.update(asdict(training_args))
    return args


def logical_axis_rules(
    activation_partitioning_dims: int = 1,
    parameter_partitioning_dims: int = 1,
    additional_rules: Optional[LogicalAxisRules] = None,
) -> LogicalAxisRules:
    """Default sharding rules in terms of logical axis names.

    Args:
      activation_partitioning_dims: enables 2-D activation sharding when set to 2.
      parameter_partitioning_dims: enables 2-D parameter sharding when set to 2.
      additional_rules: additional rules (a sequence of tuples) that will be
        appended to the standard rules.

    Returns:
      Sequence of logical axis rules
    """
    logging.info(
        "`activation_partitioning_dims` = %d, `parameter_partitioning_dims` = %d",
        activation_partitioning_dims,
        parameter_partitioning_dims,
    )

    if activation_partitioning_dims == 1 and parameter_partitioning_dims == 1:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("embed", None),
            ("image_embed", None),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("image_embed_proj", "model"),
            ("heads", "model"),
            ("image_heads", "model"),
            ("kv", None),
            ("image_kv", None),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 1:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("image_embed_proj", "model"),
            ("heads", "model"),
            ("image_heads", "model"),
            ("kv", None),
            ("image_kv", None),
            ("embed", "model"),
            ("image_embed", "model"),
        ]
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("image_embed_proj", "model"),
            ("heads", "model"),
            ("image_heads", "model"),
            ("kv", None),
            ("image_kv", None),
            ("embed", "data"),
            ("image_embed", "data"),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("image_embed_proj", "model"),
            ("heads", "model"),
            ("image_heads", "model"),
            ("kv", None),
            ("image_kv", None),
            ("joined_kv", "model"),
            ("embed", "model"),
            ("embed", "data"),
            ("image_embed", "model"),
            ("image_embed", "data"),
        ]
    else:
        raise ValueError(
            f"`activation_partitioning_dims` = {activation_partitioning_dims} "
            f"`parameter_partitioning_dims` = {parameter_partitioning_dims} "
            "is not supported."
        )

    # Add the common rules for the replicated logical axes names.
    replicated_rules = [
        ("relpos_buckets", None),
        ("abspos_buckets", None),
        ("length", None),
        ("layer", None),
        ("stack", None),
        ("mlp_activations", None),
        # U-net specific
        ("conv_height", None),
        ("conv_width", None),
        ("image_channels", None),
    ]
    rules.extend(replicated_rules)

    if additional_rules:
        rules.extend(additional_rules)

    return rules


def split_scanned_params_fn(data, merge_encoder_unet=False):
    """Split params between scanned and non-scanned"""
    # NOTE: technically this is not needed with Adam (and could be slower) but is required with shampoo
    flat = flatten_dict(data, sep=":")
    split = {"unet": {}, "encoder": {}}
    encoder_keys = ["encoder", "encoder_hidden_states_norm", "encoder_hidden_states_proj"]
    for k, v in flat.items():
        is_encoder = merge_encoder_unet or any([e in k for e in encoder_keys])
        if "blocks" in k:
            parts = k.split(":")
            shape = None
            for p in parts:
                if p.startswith("blocks"):
                    shape = p.split("blocks.")[-1]
                    break
            assert shape is not None, f"Could not find shape in {k}"
            # double check
            if hasattr(v, "value") and hasattr(v.value, "shape"):
                assert str(v.value.shape[0]) == str(shape), f"Shape mismatch: {v.value.shape[0]} != {shape}"
            group = f'scanned_{"encoder" if is_encoder else "unet"}_{shape}'
            if group not in split:
                split[group] = {}
            split[group][k] = v
        elif is_encoder:
            split["encoder"][k] = v
        else:
            split["unet"][k] = v
    # remove empty keys
    split = {k: v for k, v in split.items() if v}
    for k, v in split.items():
        split[k] = unflatten_dict(v, sep=":")
    return split


def scanned_params_bool(data, is_reshaped=False):
    """Get pytree of booleans indicating scanned layers"""
    if not is_reshaped:
        flat = flatten_dict(data)
        scanned = {}
        for k, v in flat.items():
            if "layers" in k:
                scanned[k] = True
            else:
                scanned[k] = False
        return unflatten_dict(scanned)
    else:
        return {k: v["scanned"] for k, v in data.items()}


def count_params(pytree):
    return sum([x.size for x in jax.tree_util.tree_leaves(pytree)])


def unbox_logicallypartioned(boxed_pytree):
    """Unboxes the flax.LogicallyPartitioned pieces

    Args:
      boxed_pytree: a pytree that includes LogicallyPartitioned
        leaves.
    Returns:
      a pytree where all all LogicallyPartitioned leaves have been unboxed.
    """
    return jax.tree_util.tree_map(
        lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
        boxed_pytree,
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
    )


def unsplit_scanned_params(data):
    """Reverse split_scanned_params"""
    flat = {}
    for k in data.keys():
        flat.update(flatten_dict(data[k]))
    return unflatten_dict(flat)


def _reshape_params(params, params_spec, map_to_none, MAX_PARAMS_PER_TENSOR, info_only=False, values_only=False):
    reshaped_params = {}
    flat_params_spec = flatten_dict(params_spec, sep=":")
    old_key = 0

    for k, v in flatten_dict(params, sep=":").items():
        value = unbox_logicallypartioned(v)
        shape = value.shape
        dtype = value.dtype
        sharding = v.names
        spec = flat_params_spec[k]
        scanned_layer = "layers" in sharding
        n_scanned = 0
        non_scanned_shape, non_scanned_spec = (), ()
        for shape_i, sharding_i, spec_i in zip(shape, sharding, spec):
            if sharding_i != "layers":
                if spec_i in map_to_none:
                    spec_i = None
                non_scanned_shape += (shape_i,)
                non_scanned_spec += (spec_i,)
            else:
                n_scanned = shape_i
        non_scanned_spec = tuple(non_scanned_spec)
        unique_key = f"{non_scanned_shape}-{non_scanned_spec}-{dtype}"

        if unique_key not in reshaped_params:
            reshaped_params[unique_key] = {
                "scanned": scanned_layer,
                "key_list": [k],
                "n_layers": [n_scanned],
                "non_scanned_spec": non_scanned_spec,
            }
            if not info_only:
                reshaped_params[unique_key]["values_to_scan"] = [(value, scanned_layer)]
        else:
            # prod of dims
            params_per_layer = np.prod(shape)
            n_layers = reshaped_params[unique_key]["n_layers"] + [
                n_scanned,
            ]
            n_layers = sum(min(n, 1) for n in n_layers)
            params_layer_total = params_per_layer * n_layers
            if params_layer_total > MAX_PARAMS_PER_TENSOR:
                # we create a new key
                reshaped_params[f"{old_key}"] = reshaped_params.pop(unique_key)
                old_key += 1
                # we consider it a new layer
                reshaped_params[unique_key] = {
                    "scanned": scanned_layer,
                    "key_list": [k],
                    "n_layers": [n_scanned],
                    "non_scanned_spec": non_scanned_spec,
                }
                if not info_only:
                    reshaped_params[unique_key]["values_to_scan"] = [(value, scanned_layer)]
            else:
                reshaped_params[unique_key]["key_list"].append(k)
                reshaped_params[unique_key]["n_layers"].append(n_scanned)
                if not info_only:
                    reshaped_params[unique_key]["values_to_scan"].append((value, scanned_layer))
                reshaped_params[unique_key]["scanned"] = True

    # enforce sharding + concat
    for k in reshaped_params.keys():
        layer = reshaped_params[k]
        expected_spec = layer["non_scanned_spec"]
        if layer["scanned"]:
            expected_spec = (None,) + expected_spec
        expected_spec = PartitionSpec(*expected_spec)
        reshaped_params[k]["spec"] = expected_spec
        if not info_only:
            values_to_scan = []
            for value, scanned in layer["values_to_scan"]:
                if layer["scanned"]:
                    if not scanned:
                        value = value[None, ...]
                else:
                    assert len(layer["values_to_scan"]) == 1
                    assert not scanned
                value_spec = expected_spec if layer["scanned"] else PartitionSpec(*layer["non_scanned_spec"])
                value = with_sharding_constraint(value, value_spec)
                values_to_scan.append(value)
            # concat
            if len(values_to_scan) > 1:
                reshaped_params[k]["value"] = jnp.concatenate(values_to_scan, axis=0)
            else:
                reshaped_params[k]["value"] = values_to_scan[0]
            reshaped_params[k]["value"] = with_sharding_constraint(layer["value"], expected_spec)
            del reshaped_params[k]["values_to_scan"]

    reshaped_info = {}
    reshaped_values = {}
    for k, v in reshaped_params.items():
        reshaped_info[k] = {key: v[key] for key in ["scanned", "key_list", "n_layers", "spec"]}
        if not info_only:
            reshaped_values[k] = v["value"]

    if values_only:
        return reshaped_values
    elif info_only:
        return reshaped_info
    else:
        return reshaped_values, reshaped_info


def unreshape_params(reshaped_values, reshaped_info, params):
    params = flatten_dict(params, sep=":")

    for k, layer in reshaped_info.items():
        if not layer["scanned"]:
            assert len(layer["key_list"]) == 1
            key = layer["key_list"][0]
            params[key] = params[key].replace_boxed(reshaped_values[k])
        else:
            start_idx = 0
            for key, n_layer in zip(layer["key_list"], layer["n_layers"]):
                n_layer_used = max(n_layer, 1)
                value = reshaped_values[k][start_idx : start_idx + n_layer_used]
                if n_layer == 0:
                    value = value[0]
                params[key] = params[key].replace_boxed(value)
                start_idx += n_layer_used
            assert start_idx == len(
                reshaped_values[k]
            ), f"{start_idx} != {reshaped_values[k].shape}, {layer['key_list']}, {layer['n_layers']}"
    return unflatten_dict(params, sep=":")


@dataclass
class State:
    step: int = 0
    opt_state_step: int = 0
    epoch: int = 0
    samples: int = 0
    time_total: float = 0.0
    time_train: float = 0.0
    time_per_train_step: float = 0.0
    time_per_eval: float = 0.0
    time_per_save: float = 0.0
    timestamp: float = field(init=False)
    offset_time: float = field(init=False)  # used to substract eval and save times

    def __post_init__(self):
        self.timestamp = time.perf_counter()
        self.offset_time = 0.0

    @classmethod
    def from_config_metadata(cls, config_metadata, restore_state):
        if config_metadata is not None:
            init_state = {
                k: config_metadata[k] for k, v in cls.__dataclass_fields__.items() if k in config_metadata and v.init
            }
        else:
            init_state = {}
        if not restore_state:
            init_state["opt_state_step"] = 0
        return cls(**init_state)

    def update(self, **kwargs):
        # update timing info
        if kwargs.get("step", 0) > self.step:
            now = time.perf_counter()
            self.time_total += now - self.timestamp
            delta_time = now - self.timestamp - self.offset_time
            self.time_train += delta_time
            self.offset_time = 0.0
            self.timestamp = now
            self.time_per_train_step = delta_time / (kwargs["step"] - self.step)
        # update state
        for k, v in kwargs.items():
            if isinstance(v, jnp.ndarray):
                v = jax.device_get(v)
                v = v.item()
            setattr(self, k, v)

    def add_time(self, key, duration):
        assert key in ["eval", "save"]
        if key == "eval":
            self.time_per_eval = duration
        elif key == "save":
            self.time_per_save = duration
        self.offset_time += duration

    def to_dict(self):
        # return only items that are not init=False
        return {
            k: v
            for k, v in asdict(self).items()
            if k in self.__dataclass_fields__ and self.__dataclass_fields__[k].init
        }

    def log(self, metrics={}):
        if jax.process_index() == 0:
            metrics = jax.device_get(metrics)
            metrics = unbox_logicallypartioned(metrics)

            log_metrics = flatten_dict({"state": self.to_dict()}, sep="/")
            for k, v in metrics.items():
                if "_norm" in k:
                    log_metrics[f"{k}/"] = v
                elif "_hist" in k:
                    v = jax.tree_util.tree_map(
                        lambda x: wandb.Histogram(np_histogram=x),
                        v,
                        is_leaf=lambda x: isinstance(x, tuple),
                    )
                    log_metrics[f"{k}/"] = v
                else:
                    log_metrics[k] = v
            wandb.log(log_metrics)


def should_stop_training(metrics):
    lr = metrics.get("train/learning_rate", None)
    if lr is not None and jax.device_get(lr) == 0:
        return True
    return False


def main():
    # Only initialize distributed training if running on multiple hosts
    if jax.process_count() > 1:
        jax.distributed.initialize()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Use jax cache
    if not training_args.no_cache:
        output_cache = "jax_cache"
        jax.config.update("jax_compilation_cache_dir", output_cache)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # Show arguments
    logger.info(f"Training/evaluation parameters:\n{pformat(asdict(training_args))}")
    logger.info(f"Model parameters:\n{pformat(asdict(model_args))}")
    logger.info(f"Data parameters:\n{pformat(asdict(data_args))}")

    # Info on local devices
    logger.info(f"Local TPUs/GPUs: {jax.local_device_count()}")
    logger.info(f"Global TPUs/GPUs: {jax.device_count()}")

    # Set up wandb run
    if jax.process_index() == 0:
        try:
            wandb.init(
                entity=training_args.wandb_entity,
                project=training_args.wandb_project,
                job_type=training_args.wandb_job_type,
                config=flat_args(model_args, data_args, training_args),
                save_code=False,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            logger.warning("Continuing without wandb logging...")
            # Create a dummy wandb.log function that does nothing
            def dummy_log(*args, **kwargs):
                pass
            wandb.log = dummy_log

    # update helper functions
    split_scanned_params = partial(
        split_scanned_params_fn,
        merge_encoder_unet=training_args.block_size_encoder == training_args.block_size_unet,
    )

    def get_opt_state_step(opt_state):
        if training_args.optim == "distributed_shampoo":
            return opt_state["unet"][0] if "unet" in opt_state else opt_state["encoder"][0]
        elif training_args.optim == "adam":
            return opt_state["unet"][2].count if "unet" in opt_state else opt_state["encoder"][2].count
        elif training_args.optim == "kron":
            psgd_idx = 0
            for part in opt_state:
                if isinstance(part, dict):  # kron state is a dict
                    if "Qs_preconditioners" in part:
                        break
                psgd_idx += 1
            return opt_state[psgd_idx]["count"]

        else:
            raise ValueError(f"Optimizer {training_args.optim} not supported")

    # Load state
    state = State.from_config_metadata(None, False)

    # Set rng
    rng = jax.random.PRNGKey(training_args.seed_model)

    with open("logical_params.pkl", "rb") as f:
        logical_params = pickle.load(f)

    # Parameter count
    num_params = {
        "Total": count_params(logical_params),
        "U-Net": count_params(logical_params["unet"]),
        "Encoder": count_params(logical_params["encoder"]),
    }
    num_hosts = jax.process_count()

    # Log some info
    logger.info(f"Num epochs: {training_args.num_train_epochs}")
    logger.info(f"Number of devices = {jax.device_count()}")
    logger.info(f"Number of nodes = {num_hosts}")
    logger.info(
        f"Model parameters: U-Net {num_params['U-Net']:,} + Encoder {num_params['Encoder']:,} = {num_params['Total']:,}"
    )

    # Partition Spec
    logical_spec = nn.get_partition_spec(logical_params)
    rules = logical_axis_rules(
        activation_partitioning_dims=training_args.activation_partitioning_dims,
        parameter_partitioning_dims=training_args.parameter_partitioning_dims,
    )
    params_spec = nn.logical_to_mesh(logical_spec, rules)
    scan_spec = PartitionSpec(None)

    # Create mesh
    logger.info(f"Creating a mesh of ({training_args.dp_devices}, {training_args.mp_devices})")
    dev_mesh = create_device_mesh((training_args.dp_devices, training_args.mp_devices))
    mesh = Mesh(dev_mesh, ("data", "model"))
    # mesh = default_mesh(training_args.mp_devices)
    logger.info(f"Mesh: {mesh.shape}")

    # Initialize or restore model
    logger.info("Initializing model parameters")

    @partial(pjit, in_shardings=(None,), out_shardings=params_spec)
    def init_params(rng):
        return jax.tree_util.tree_map(lambda x: jax.random.normal(rng, x.shape, dtype=x.dtype), logical_params)

    # Set params
    params_rng, rng = jax.random.split(rng)
    with mesh:
        params = init_params(params_rng)


    def trainable_params(data):
        valid_keys = None
        if training_args.trainable_params_mode == "encoder_warmup":
            valid_keys = ["encoder"]
        if valid_keys is not None:
            data_train = {}
            for k, v in flatten_dict(data, sep=":").items():
                if any([e in k for e in valid_keys]):
                    data_train[k] = v
            return unflatten_dict(data_train, sep=":")
        else:
            return data

    # Create learning rate schedule
    def create_learning_rate_fn() -> Callable[[int], jnp.array]:
        """Create the learning rate function."""

        def _add_schedule(schedule, new_schedule, boundary):
            if schedule is None:
                return new_schedule
            else:
                return optax.join_schedules(
                    schedules=[schedule, new_schedule],
                    boundaries=[boundary],
                )

        # build schedule
        schedule_fn = None
        last_boundary = 0

        # offset
        lr_offset = training_args.lr_offset + state.opt_state_step
        if lr_offset:
            schedule_fn = _add_schedule(schedule_fn, optax.constant_schedule(0.0), last_boundary)
            last_boundary += lr_offset

        # warmup
        if training_args.warmup_steps > 0:
            new_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=training_args.learning_rate,
                transition_steps=training_args.warmup_steps,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
            last_boundary += training_args.warmup_steps

        # decay
        if training_args.lr_decay == "linear":
            new_schedule = optax.linear_schedule(
                init_value=training_args.learning_rate,
                end_value=0,
                transition_steps=training_args.lr_transition_steps,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        elif training_args.lr_decay == "exponential":
            new_schedule = optax.exponential_decay(
                init_value=training_args.learning_rate,
                transition_steps=training_args.lr_transition_steps,
                decay_rate=training_args.lr_decay_rate,
                staircase=training_args.lr_staircase,
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        elif training_args.lr_decay == "cosine":
            new_schedule = optax.cosine_decay_schedule(
                init_value=training_args.learning_rate, decay_steps=training_args.lr_transition_steps
            )
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)
        else:
            # constant
            new_schedule = optax.constant_schedule(training_args.learning_rate)
            schedule_fn = _add_schedule(schedule_fn, new_schedule, last_boundary)

        return schedule_fn

    learning_rate_fn = create_learning_rate_fn()

    # compact args
    # TODO: could use this for shampoo too
    if training_args.compact_params:
        axis_to_n_devices = {axis: n for axis, n in zip(mesh.axis_names, mesh.devices.shape)}
        map_to_none = [k for k, n in axis_to_n_devices.items() if n == 1]
        print(f"map_to_none: {map_to_none}")
        reshape_params = partial(
            _reshape_params, map_to_none=map_to_none, MAX_PARAMS_PER_TENSOR=training_args.compact_params
        )
        reshaped_info = reshape_params(trainable_params(params), trainable_params(params_spec), info_only=True)
        with mesh:
            logical_params_for_opt = jax.eval_shape(
                lambda x: reshape_params(
                    trainable_params(x), trainable_params(params_spec), info_only=False, values_only=True
                ),
                params,
            )
        params_spec_for_opt = {k: v["spec"] for k, v in reshaped_info.items()}
        scanned_layers_arg = scanned_params_bool(reshaped_info, is_reshaped=True)

        print(f"Number of tensors in reshaped params: {len(reshaped_info)}")
        for k, v in reshaped_info.items():
            l_shape = logical_params_for_opt[k].shape
            l_params = np.prod(l_shape)
            print(f"{l_shape} - {v['spec']} {'(scanned) ' if scanned_layers_arg[k] else ''}- {l_params:,} params")
        print()
    else:
        logical_params_for_opt = trainable_params(logical_params)
        params_spec_for_opt = trainable_params(params_spec)
        scanned_layers_arg = scanned_params_bool(logical_params_for_opt, is_reshaped=False)

    # create optimizer
    if training_args.optim == "distributed_shampoo":
        graft_type = {
            "sgd": GraftingType.SGD,
            "adagrad": GraftingType.ADAGRAD,
            "rmsprop": GraftingType.RMSPROP,
            "rmsprop_normalized": GraftingType.RMSPROP_NORMALIZED,
            "sqrt_n": GraftingType.SQRT_N,
            "adagrad_normalized": GraftingType.ADAGRAD_NORMALIZED,
        }[training_args.graft_type]
        statistics_partition_spec = (
            PartitionSpec(None, training_args.shard_shampoo_across, None)
            if training_args.shard_shampoo_across != "2d"
            else PartitionSpec(None, "data", "model")
        )
        preconditioner_axis = (
            training_args.shard_shampoo_across
            if training_args.shard_shampoo_across != "2d"
            else "model"
            if training_args.mp_devices > training_args.dp_devices
            else "data"
        )
        preconditioner_num_devices = (
            training_args.mp_devices if preconditioner_axis == "model" else training_args.dp_devices
        )
        _opt = partial(
            distributed_shampoo,
            beta1=training_args.beta1,
            beta2=training_args.beta2,
            diagonal_epsilon=1e-10,
            matrix_epsilon=1e-6,
            weight_decay=training_args.weight_decay,
            start_preconditioning_step=max(training_args.preconditioning_compute_steps + 1, 101),
            caspr_variant=training_args.caspr_variant,
            preconditioning_compute_steps=training_args.preconditioning_compute_steps,
            statistics_compute_steps=1,
            best_effort_shape_interpretation=True,
            graft_type=graft_type,
            nesterov=training_args.nesterov,
            exponent_override=training_args.exponent_override,
            statistics_partition_spec=statistics_partition_spec,
            preconditioner_partition_spec=PartitionSpec(preconditioner_axis, None, None),
            num_devices_for_pjit=preconditioner_num_devices,
            shard_optimizer_states=True,
            inverse_failure_threshold=0.1,
            moving_average_for_momentum=True,
            skip_preconditioning_dim_size_gt=training_args.skip_preconditioning_dim_size_gt,
            clip_by_scaled_gradient_norm=training_args.clip_by_scaled_gradient_norm,
            precision=jax.lax.Precision.HIGHEST,
            best_effort_memory_usage_reduction=training_args.optim_quantized,
            eigh=training_args.eigh,
        )
        # get the real optimizer and helper functions
        opt_encoder = _opt(learning_rate_fn, block_size=training_args.block_size_encoder)
        opt_unet = _opt(learning_rate_fn, block_size=training_args.block_size_unet)
        update_fn_encoder = opt_encoder.update
        update_fn_unet = opt_unet.update
        # we need to allow scanned layers
        optimizer = {}
        opt_fn = {}
        for k, p in split_scanned_params(trainable_params(logical_params)).items():
            if "scanned" in k:
                # extract 1 layer shape by removing first dimension
                p_shape = jax.tree_util.tree_map(
                    lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), 
                    p
                )
                p = p_shape
            optimizer[k] = opt_unet.init(p) if "unet" in k else opt_encoder.init(p)
            opt_fn[k] = NamedTuple("opt_fn", pspec_fn=Any, shape_and_dtype_fn=Any)(
                optimizer[k].pspec_fn, optimizer[k].shape_and_dtype_fn
            )
            optimizer[k] = optax.GradientTransformation(
                optimizer[k].init_fn, update_fn_unet if "unet" in k else update_fn_encoder
            )
    elif training_args.optim == "kron":
        # psgd kron handles scanned layers internally so we pass in a tree of booleans
        # indicating which layers to scan
        _opt = [
            optax.clip_by_global_norm(1.0),
            scale_by_kron(
                b1=training_args.beta1,
                preconditioner_update_probability=precond_update_prob_schedule(
                    min_prob=1 / training_args.preconditioning_compute_steps,
                ),
                max_size_triangular=training_args.skip_preconditioning_dim_size_gt,
                memory_save_mode=training_args.kron_mem_save_mode,
                merge_small_dims=training_args.kron_merge_small_dims,
                scanned_layers=scanned_layers_arg,
                lax_map_scanned_layers=False,
                lax_map_batch_size=8,
            ),
        ]
        if training_args.weight_decay > 0:
            _opt.append(optax.add_decayed_weights(training_args.weight_decay))
        _opt.append(optax.scale_by_learning_rate(learning_rate_fn))
        optimizer = optax.chain(*_opt)

    elif training_args.optim == "adam":
        _opt = partial(
            optax.adamw,
            b1=training_args.beta1,
            b2=training_args.beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        optimizer = {
            k: _opt(learning_rate=learning_rate_fn) for k in split_scanned_params(trainable_params(logical_params))
        }

    # get PartitionSpec of optimizer state
    def get_opt_state_spec_psgd():
        # Create dummy arrays for optimizer init
        logical_params_for_opt = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype),
            logical_params
        )
        
        # Initialize optimizer with dummy arrays
        opt_state = optimizer.init(logical_params_for_opt)
        
        # Convert optimizer state to specs
        def _to_spec(x):
            if isinstance(x, (jax.ShapeDtypeStruct, optax.EmptyState)):
                return None
            return PartitionSpec(None) if x.ndim > 0 else PartitionSpec()
        
        opt_state_spec = jax.tree_util.tree_map(
            _to_spec,
            opt_state,
            is_leaf=lambda x: isinstance(x, (jax.ShapeDtypeStruct, optax.EmptyState))
        )
        
        return opt_state_spec

    def get_opt_state_spec():
        # get opt_state shape without actual init
        opt_state_shape = {}
        for k, p in split_scanned_params(trainable_params(logical_params)).items():
            if "scanned" in k:
                # Get the scan dimension from any parameter in p
                scan_dim = jax.tree_util.tree_leaves(p)[0].shape[0]
                
                # Create shape info for a single layer by removing first dimension
                p_shape = jax.tree_util.tree_map(
                    lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype), 
                    p
                )
                # Create dummy array for single layer
                p_dummy = jax.tree_util.tree_map(
                    lambda x: jnp.zeros(x.shape, x.dtype),
                    p_shape
                )
                # Get init shape for single layer
                single_shape = optimizer[k].init(p_dummy)
                # Add back scan dimension
                opt_state_shape[k] = jax.tree_util.tree_map(
                    lambda x: jax.ShapeDtypeStruct((scan_dim,) + x.shape, x.dtype),
                    single_shape,
                    is_leaf=lambda x: isinstance(x, (jax.ShapeDtypeStruct, optax.EmptyState))
                )
            else:
                # For non-scanned layers, create dummy array
                p_dummy = jax.tree_util.tree_map(
                    lambda x: jnp.zeros(x.shape, x.dtype),
                    p
                )
                opt_state_shape[k] = optimizer[k].init(p_dummy)

        # utility functions for Adam
        def _adam_opt_state_spec_per_leaf(x, spec):
            if isinstance(x, (jax.ShapeDtypeStruct, optax.EmptyState)):
                return spec
            else:
                return None

        def _adam_pspec_fn(spec, shape):
            return (
                None
                if spec is None
                else jax.tree_util.tree_map(
                    partial(_adam_opt_state_spec_per_leaf, spec=spec),
                    shape,
                    is_leaf=lambda x: isinstance(x, (jax.ShapeDtypeStruct, optax.EmptyState))
                )
            )

        # get PartitionSpec
        split_spec = split_scanned_params(trainable_params(params_spec))
        opt_state_spec = {}

        def _get_spec(**kwargs):
            """Get optimizer spec for a certain model portion"""
            if training_args.optim == "adam":
                return _adam_pspec_fn(kwargs["params_spec"], kwargs["opt_state_shape"])
            elif training_args.optim == "distributed_shampoo":
                return kwargs["opt_fn"].pspec_fn(
                    kwargs["params_shape"],
                    kwargs["params_spec"],
                    statistics_partition_spec,
                )
            else:
                raise NotImplementedError

        for k, p in split_scanned_params(trainable_params(logical_params)).items():
            p_spec = split_spec[k]
            if "scanned" in k:
                # extract 1 layer shape
                p_shape = jax.tree_util.tree_map(
                    lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
                    p
                )
                p_spec = jax.tree_util.tree_map(
                    lambda x: PartitionSpec(*x[1:]) if x is not None else None,
                    p_spec
                )
            _opt_fn = opt_fn[k] if training_args.optim == "distributed_shampoo" else None
            opt_state_spec[k] = _get_spec(
                params_spec=p_spec,
                opt_state_shape=opt_state_shape[k],
                opt_fn=_opt_fn,
                params_shape=p_shape if "scanned" in k else p,
            )
            if "scanned" in k:
                # add scan dimension
                opt_state_spec[k] = jax.tree_util.tree_map(
                    lambda x: PartitionSpec(*scan_spec + x) if x is not None else None,
                    opt_state_spec[k],
                    is_leaf=lambda x: isinstance(x, (PartitionSpec, type(None)))
                )

        return opt_state_spec

    if training_args.optim == "kron":
        opt_state_spec = get_opt_state_spec_psgd()
    else:
        opt_state_spec = get_opt_state_spec()

    # Initialize or restore optimizer state
    logger.info("Initializing optimizer state")

    @partial(pjit, in_shardings=(params_spec,), out_shardings=opt_state_spec)
    def init_opt_state(params):
        if training_args.optim == "kron":
            if training_args.compact_params:
                used_params = reshape_params(trainable_params(params), trainable_params(params_spec), values_only=True)
            else:
                used_params = trainable_params(params)
            return optimizer.init(used_params)
        else:
            opt_state = {}
            for k, p in split_scanned_params(trainable_params(params)).items():
                init_fn = optimizer[k].init
                if "scanned" in k:
                    init_fn = jax.vmap(init_fn)
                opt_state[k] = init_fn(p)
            return opt_state

    # Set opt_state
    with mesh:
        opt_state = init_opt_state(params)

    # Define update function
    def update_params(params, opt_state, grads):
        if training_args.optim == "kron":
            grads = trainable_params(grads)
            new_params = trainable_params(params)
            if training_args.compact_params:
                new_params, reshaped_info = reshape_params(
                    new_params, trainable_params(params_spec), info_only=False, values_only=False
                )
                grads, _ = reshape_params(grads, trainable_params(params_spec), info_only=False)
            updates, new_opt_state = optimizer.update(grads, opt_state, new_params)
            new_params = optax.apply_updates(new_params, updates)
            if training_args.compact_params:
                new_params = unreshape_params(new_params, reshaped_info, trainable_params(params))
        else:
            grads = split_scanned_params(trainable_params(grads))
            split_params = split_scanned_params(trainable_params(params))
            new_opt_state = {}
            new_params = {}
            for k, param in split_params.items():
                update_fn = optimizer[k].update
                if "scanned" in k:
                    update_fn = jax.vmap(update_fn, in_axes=(0, 0, 0), out_axes=(0, 0))
                updates, new_opt_state[k] = update_fn(grads[k], opt_state[k], param)
                new_params[k] = optax.apply_updates(param, updates)
            new_params = unsplit_scanned_params(new_params)
        # merge with non-trainable params
        params, new_params = flatten_dict(params), flatten_dict(new_params)
        params.update(new_params)
        new_params = unflatten_dict(params)

        return new_params, new_opt_state

    # Training step
    @partial(
        pjit,
        in_shardings=(None, params_spec),
        out_shardings=(params_spec, None),
    )
    def compute_grads(rng, params):
        # NOTE: fake grads
        metrics = {"train/loss": 1.0}
        grads = jax.tree_util.tree_map(lambda x: jax.random.normal(rng, x.shape, dtype=x.dtype), params)
        # normalize grads
        grads = jax.tree_util.tree_map(lambda x: x / jnp.linalg.norm(x), grads)
        return grads, metrics

    @partial(
        pjit,
        in_shardings=(
            params_spec,
            opt_state_spec,
            params_spec,
            None,
        ),
        out_shardings=(
            None,
            params_spec,
            opt_state_spec,
            None,
        ),
        donate_argnums=(0, 1, 2, 3),
    )
    def apply_grads(params, opt_state, grads, step):
        print("Applying gradients")
        # update params
        new_params, new_opt_state = update_params(params, opt_state, grads)

        # get opt_state_step
        opt_state_step = get_opt_state_step(new_opt_state)

        # metrics
        metrics = {}

        # learning rate
        metrics["train/learning_rate"] = learning_rate_fn(opt_state_step)

        # check params are finite
        if training_args.skip_update:
            # update
            require_finite_all = True  # TODO: hardcoded - whether we require all params to be finite
            if require_finite_all:
                finite_all = jnp.all(
                    jnp.asarray([jnp.all(jnp.isfinite(p)) for p in jax.tree_util.tree_leaves(new_params)])
                )
                new_params = jax.tree.map(lambda a, b: jnp.where(finite_all, a, b), new_params, params)
                new_opt_state = jax.tree.map(lambda a, b: jnp.where(finite_all, a, b), new_opt_state, opt_state)
                #  metrics = jax.tree.map(lambda a: jnp.where(finite_all, a, 0.0), metrics)
            else:
                new_params = jax.tree.map(lambda a, b: jnp.where(jnp.isfinite(a), a, b), new_params, params)
                new_opt_state = jax.tree.map(lambda a, b: jnp.where(jnp.isfinite(a), a, b), new_opt_state, opt_state)
                # metrics = jax.tree.map(lambda a: jnp.where(jnp.isfinite(a), a, 0.0), metrics)

        # increment step
        step += 1

        # extract norms and histograms
        def maybe_fn(fn, val, freq):
            """Call fn only if it is a logging step - NOTE: is it faster?"""
            trainable_val = trainable_params(val)
            return jax.lax.cond(
                step % freq == 0,
                fn,
                lambda p: jax.tree.map(lambda v: jnp.zeros_like(v), fn(p)),
                trainable_val,
            )

        if training_args.log_norm_steps:

            def norm(val):
                return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), val)

            gradients_norm = maybe_fn(norm, grads, training_args.log_norm_steps)
            params_norm = maybe_fn(norm, new_params, training_args.log_norm_steps)
            metrics.update(
                {
                    "gradients_norm": gradients_norm,
                    "params_norm": params_norm,
                }
            )

        if training_args.log_histogram_steps:

            def histogram(val):
                return jax.tree_util.tree_map(lambda x: jnp.histogram(x, density=True), val)

            gradients_hist = maybe_fn(histogram, grads, training_args.log_histogram_steps)
            params_hist = maybe_fn(histogram, new_params, training_args.log_histogram_steps)

            metrics.update(
                {
                    "params_hist": params_hist,
                    "gradients_hist": gradients_hist,
                }
            )

        return metrics, new_params, new_opt_state, opt_state_step

    # Init training variables
    has_compiled, metrics_logged, stop_training = (
        False,
        False,
        False,
    )
    step, samples = state.step, state.samples  # separate copies for timing metrics
    opt_state_step = 0  # ensure it is defined in evaluation mode
    step_start = step  # define it for test mode
    metrics_loss = {}
    metrics_grads = {}
    epochs = tqdm(
        range(training_args.num_train_epochs), desc=f"Epoch ... (1/{training_args.num_train_epochs})", position=0
    )

    # Training loop
    logger.info("***** Running training *****")
    for epoch in epochs:
        if stop_training:
            break
        state.update(epoch=epoch)
        state.log({})

        # Training
        if training_args.do_train:
            dataset = list(range(100))
            for _ in tqdm(dataset, desc="Training...", position=1, leave=False, disable=training_args.debug):
                if training_args.debug:
                    print(f"Training step: {step}")
                    multihost_utils.sync_global_devices(f"training step {step}")

                # reset control variables
                metrics_logged = False

                # start profile
                if training_args.do_profile:
                    if step == training_args.start_profile:
                        jax.block_until_ready(params)
                        jax.profiler.start_trace("./profiles")

                # train step
                step_rng, rng = jax.random.split(rng)
                if training_args.debug:
                    print("Performing step...")
                with mesh:
                    # compute grads
                    if not has_compiled and training_args.do_lower:
                        logger.info("Lowering compute_grads...")
                        compute_grads = compute_grads.lower(step_rng, params)
                        lowered_compute_grads = compute_grads.as_text()
                        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        with open(f"compute_grads_lowered_{timestamp}.txt", "w") as f:
                            f.write(lowered_compute_grads)
                        logger.info("Compiling compute_grads...")
                        compute_grads = compute_grads.compile()
                        logger.info("Compiled compute_grads")
                    grads, metrics_loss = compute_grads(
                        step_rng,
                        params,
                    )

                    # apply grads
                    if not has_compiled and training_args.do_lower:
                        logger.info("Lowering apply_grads...")
                        apply_grads = apply_grads.lower(params, opt_state, grads, step)
                        lowered_apply_grads = apply_grads.as_text()
                        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        with open(f"apply_grads_lowered_{timestamp}.txt", "w") as f:
                            f.write(lowered_apply_grads)
                        logger.info("Compiling apply_grads...")
                        apply_grads = apply_grads.compile()
                        logger.info("Compiled apply_grads")
                        has_compiled = True
                    metrics_grads, params, opt_state, opt_state_step = apply_grads(params, opt_state, grads, step)
                    del grads

                step += 1

                # end profile
                if training_args.do_profile:
                    if (step - 1) == training_args.end_profile:
                        jax.block_until_ready(params)
                        jax.profiler.stop_trace()

                # log metrics
                if step % training_args.logging_steps == 0:
                    state.update(step=step, samples=samples, opt_state_step=opt_state_step)
                    metrics = {**metrics_loss, **metrics_grads}
                    if training_args.debug:
                        print(metrics)
                    if jax.process_index() == 0:
                        state.log(metrics)
                    metrics_logged = True
                    stop_training = should_stop_training(metrics)

                # end test
                if training_args.do_test_steps and (step - step_start >= training_args.do_test_steps):
                    # terminate script
                    print("Test successful")
                    return

                # end training
                if stop_training:
                    break

    # log final metrics
    if not metrics_logged:
        state.update(step=step, samples=samples, opt_state_step=opt_state_step)
        metrics = {**metrics_loss, **metrics_grads}
        if jax.process_index() == 0:
            state.log(metrics)


if __name__ == "__main__":
    main()
