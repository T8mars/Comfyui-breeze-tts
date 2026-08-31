"""INT8 ConvRot runtime for Breeze TTS 2 (comfy-kitchen).

Quantized checkpoints store, per quantized linear:
  weight       : torch.int8, offline-rotated per group (W_rot = W @ H^T)
  weight_scale : float32 [out_features, 1] per-output-row scale
  comfy_quant  : uint8 JSON {"format": "int8_tensorwise", "convrot": true,
                             "convrot_groupsize": G}
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger("BreezeTTS2")

QUANT_META_SUFFIX = "comfy_quant"
SUPPORTED_FORMAT = "int8_tensorwise"

RUNTIME_STATS: dict = {"calls": 0, "modules": 0, "weight_dtype": None, "groupsize": None}


def validate_group_size(group_size: int, in_features: int) -> None:
    if group_size < 4 or group_size & (group_size - 1) != 0 or math.log(group_size, 4) % 1 != 0:
        raise ValueError(f"ConvRot group size must be a power of four (4/16/64/256/...), got {group_size}")
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} is not divisible by group size {group_size}")


@dataclass
class QuantLayerInfo:
    prefix: str
    group_size: int
    in_features: int = 0
    out_features: int = 0
    has_bias: bool = False


def _shard_files(checkpoint) -> list:
    from pathlib import Path

    path = Path(checkpoint)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            return [path / shard for shard in sorted(set(weight_map.values()))]
        return [path / "model.safetensors"]
    return [path]


def scan_checkpoint_quantization(checkpoint) -> dict[str, QuantLayerInfo]:
    """Read every *.comfy_quant key in the checkpoint.

    Returns {} for a plain float checkpoint. Raises on quant formats this
    nodepack cannot execute so INT8 weights are never silently misread as
    floats.
    """
    from safetensors import safe_open

    quant_map: dict[str, QuantLayerInfo] = {}
    for shard in _shard_files(checkpoint):
        if not shard.is_file():
            continue
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            key_names = list(f.keys())
            for key in [k for k in key_names if k.endswith(f".{QUANT_META_SUFFIX}")]:
                meta = json.loads(f.get_tensor(key).numpy().tobytes())
                if meta.get("format") != SUPPORTED_FORMAT or meta.get("convrot") is not True:
                    raise RuntimeError(
                        f"{shard.name}:{key} uses quant format {meta!r}; only "
                        f"'{SUPPORTED_FORMAT}' with convrot=true is supported."
                    )
                prefix = key[: -len(f".{QUANT_META_SUFFIX}")]
                quant_map[prefix] = QuantLayerInfo(
                    prefix=prefix,
                    group_size=int(meta["convrot_groupsize"]),
                    in_features=int(meta.get("in_features", 0)),
                    out_features=int(meta.get("out_features", 0)),
                    has_bias=bool(meta.get("has_bias", f"{prefix}.bias" in key_names)),
                )
    return quant_map


class ConvRotInt8Linear(nn.Module):
    """Drop-in nn.Linear replacement executing the comfy-kitchen INT8 ConvRot path."""

    def __init__(self, in_features: int, out_features: int, bias: bool, group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(out_features, 1, dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None
        self.quant_format = SUPPORTED_FORMAT

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict.pop(f"{prefix}{QUANT_META_SUFFIX}", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        import comfy_kitchen

        RUNTIME_STATS["calls"] += 1
        return comfy_kitchen.int8_linear(
            x.contiguous(),
            self.weight,
            self.weight_scale,
            self.bias,
            out_dtype=x.dtype,
            convrot=True,
            convrot_groupsize=self.convrot_groupsize,
        )


def replace_quantized_linears(model: nn.Module, quant_map: dict[str, QuantLayerInfo]) -> list[str]:
    """Swap every nn.Linear named in quant_map for a ConvRotInt8Linear.

    Must run BEFORE weights are assigned. Raises if a target is missing, is
    not an nn.Linear, or its shape disagrees with the recorded metadata.
    """
    modules = dict(model.named_modules())
    replaced: list[str] = []
    for prefix, info in quant_map.items():
        target = modules.get(prefix)
        if target is None:
            raise RuntimeError(
                f"Quantized layer '{prefix}' not found in the model. The checkpoint "
                "does not match this model build."
            )
        if isinstance(target, ConvRotInt8Linear):
            replaced.append(prefix)
            continue
        if not isinstance(target, nn.Linear):
            raise RuntimeError(f"Quantized layer '{prefix}' is {type(target).__name__}, expected nn.Linear.")
        if info.in_features and target.in_features != info.in_features:
            raise RuntimeError(
                f"Quantized layer '{prefix}' in_features mismatch: checkpoint {info.in_features}, model {target.in_features}."
            )
        if info.out_features and target.out_features != info.out_features:
            raise RuntimeError(
                f"Quantized layer '{prefix}' out_features mismatch: checkpoint {info.out_features}, model {target.out_features}."
            )
        validate_group_size(info.group_size, target.in_features)
        parent_name, _, child_name = prefix.rpartition(".")
        setattr(modules[parent_name] if parent_name else model, child_name,
                ConvRotInt8Linear(target.in_features, target.out_features, target.bias is not None, info.group_size))
        replaced.append(prefix)
    return replaced


def quantized_parameter_count(model: nn.Module) -> tuple[int, int]:
    params = sum(
        m.weight.numel() for m in model.modules() if isinstance(m, ConvRotInt8Linear)
    )
    modules = sum(1 for m in model.modules() if isinstance(m, ConvRotInt8Linear))
    return params, modules


def log_int8_banner(model: nn.Module, device: torch.device) -> None:
    import comfy_kitchen
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    qparams, qcount = quantized_parameter_count(model)
    group_sizes = sorted({m.convrot_groupsize for m in model.modules() if isinstance(m, ConvRotInt8Linear)})
    smoke = "SKIPPED (cpu)"
    if device.type in ("cuda", "xpu"):
        g = group_sizes[0]
        w = torch.randn(64, g * 2, device=device)
        q, p = TensorWiseINT8Layout.quantize(
            w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=g, stochastic_rounding=0
        )
        y = comfy_kitchen.int8_linear(
            torch.randn(8, g * 2, device=device), q, p.scale, None,
            out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=g,
        )
        smoke = "PASS" if bool(torch.isfinite(y).all()) else "FAIL"
    backends = ",".join(sorted(k for k, v in comfy_kitchen.list_backends().items() if v["available"]))
    RUNTIME_STATS.update(modules=qcount, weight_dtype=torch.int8, groupsize=group_sizes[0] if group_sizes else None)
    logger.info(
        "INT8 ConvRot: %d layers / %.2fB params / group %s / comfy_kitchen.int8_linear / backends %s / runtime smoke %s",
        qcount, qparams / 1e9, group_sizes, backends, smoke,
    )
    if smoke.startswith("FAIL"):
        raise RuntimeError("comfy_kitchen INT8 ConvRot kernel smoke test failed; refusing to run the quantized model.")
