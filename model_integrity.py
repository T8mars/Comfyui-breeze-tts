"""Fast, dependency-free integrity checks for a local Breeze TTS 2 snapshot.

The checks intentionally read only JSON files and safetensors headers.  They
therefore catch interrupted/truncated downloads without hashing or mapping the
multi-gigabyte tensor payloads on every ComfyUI queue run.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REQUIRED_JSON_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "audio_tokenizer/config.json",
)
REQUIRED_CODEC_WEIGHTS = "audio_tokenizer/model.safetensors"
MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ModelIntegrityReport:
    model_dir: Path
    missing_files: tuple[str, ...]
    invalid_files: tuple[str, ...]
    referenced_shards: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_files and not self.invalid_files and bool(self.referenced_shards)

    def summary(self) -> str:
        details: list[str] = []
        if self.missing_files:
            details.append("缺少: " + ", ".join(self.missing_files))
        if self.invalid_files:
            details.append("损坏/无效: " + "; ".join(self.invalid_files))
        if not details:
            return "模型文件完整"
        return "；".join(details)


def _load_json_object(path: Path) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{path.name}: JSON 无法读取 ({exc})"
    if not isinstance(value, dict) or not value:
        return None, f"{path.name}: JSON 内容为空或不是对象"
    return value, None


def _safe_snapshot_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        return None
    return relative.as_posix()


def inspect_safetensors_header(path: Path) -> tuple[frozenset[str] | None, str | None]:
    """Validate safetensors layout using only its header and the file size."""

    try:
        file_size = path.stat().st_size
        if file_size <= 8:
            return None, "文件过小"
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                return None, "缺少 safetensors 头"
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                return None, f"头长度异常 ({header_size})"
            if 8 + header_size > file_size:
                return None, "文件在 safetensors 头内被截断"
            header = json.loads(handle.read(header_size).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
        return None, f"无法读取 safetensors 头 ({exc})"

    if not isinstance(header, dict):
        return None, "safetensors 头不是对象"
    tensor_names: set[str] = set()
    max_end = 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            return None, "张量目录格式无效"
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            return None, f"张量 {name!r} 的 data_offsets 无效"
        tensor_names.add(name)
        max_end = max(max_end, offsets[1])
    if not tensor_names:
        return None, "未包含任何张量"
    payload_size = file_size - 8 - header_size
    if max_end != payload_size:
        return None, f"张量数据长度不符 (期望 {max_end}，实际 {payload_size})"
    return frozenset(tensor_names), None


def inspect_model_dir(model_dir: Path, weights_name: str) -> ModelIntegrityReport:
    """Return every missing/invalid required file for one local snapshot."""

    model_dir = Path(model_dir)
    missing: list[str] = []
    invalid: list[str] = []

    for relative in REQUIRED_JSON_FILES:
        path = model_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        _, issue = _load_json_object(path)
        if issue:
            invalid.append(f"{relative} ({issue})")

    codec_path = model_dir / REQUIRED_CODEC_WEIGHTS
    if not codec_path.is_file():
        missing.append(REQUIRED_CODEC_WEIGHTS)
    else:
        _, issue = inspect_safetensors_header(codec_path)
        if issue:
            invalid.append(f"{REQUIRED_CODEC_WEIGHTS} ({issue})")

    index_path = model_dir / weights_name
    referenced_shards: list[str] = []
    weight_map: dict[str, object] = {}
    if not index_path.is_file():
        missing.append(weights_name)
    else:
        index, issue = _load_json_object(index_path)
        if issue:
            invalid.append(f"{weights_name} ({issue})")
        else:
            raw_map = index.get("weight_map")
            if not isinstance(raw_map, dict) or not raw_map:
                invalid.append(f"{weights_name} (缺少非空 weight_map)")
            else:
                weight_map = raw_map
                bad_values = sorted(
                    {str(value) for value in raw_map.values() if _safe_snapshot_relative_path(value) is None}
                )
                if bad_values:
                    invalid.append(f"{weights_name} (包含不安全的分片路径: {', '.join(bad_values)})")
                else:
                    referenced_shards = sorted({_safe_snapshot_relative_path(value) for value in raw_map.values()})

    shard_headers: dict[str, frozenset[str]] = {}
    for relative in referenced_shards:
        shard_path = model_dir / relative
        if not shard_path.is_file():
            missing.append(relative)
            continue
        keys, issue = inspect_safetensors_header(shard_path)
        if issue:
            invalid.append(f"{relative} ({issue})")
        elif keys is not None:
            shard_headers[relative] = keys

    # An index can point a tensor at an existing but wrong shard. Catch that
    # cheaply by comparing weight_map entries with each shard's header keys.
    missing_indexed_tensors: list[str] = []
    if referenced_shards and shard_headers:
        for tensor_name, raw_shard in weight_map.items():
            relative = _safe_snapshot_relative_path(raw_shard)
            if relative in shard_headers and tensor_name not in shard_headers[relative]:
                missing_indexed_tensors.append(str(tensor_name))
    if missing_indexed_tensors:
        preview = ", ".join(missing_indexed_tensors[:5])
        suffix = " …" if len(missing_indexed_tensors) > 5 else ""
        invalid.append(f"{weights_name} (索引张量未出现在对应分片: {preview}{suffix})")

    return ModelIntegrityReport(
        model_dir=model_dir,
        missing_files=tuple(sorted(set(missing))),
        invalid_files=tuple(invalid),
        referenced_shards=tuple(referenced_shards),
    )


def repair_guidance(report: ModelIntegrityReport) -> str:
    return (
        f"{report.summary()}。模型目录: {report.model_dir}。"
        "请在 T8 模型加载器中保持 download_if_missing=true 后重新执行，节点会从固定 revision 续传。"
        "若同一损坏文件反复出现，请关闭 ComfyUI，仅删除上面列出的损坏文件后重试；不要删除其他模型。"
    )


__all__ = [
    "ModelIntegrityReport",
    "REQUIRED_CODEC_WEIGHTS",
    "REQUIRED_JSON_FILES",
    "inspect_model_dir",
    "inspect_safetensors_header",
    "repair_guidance",
]
