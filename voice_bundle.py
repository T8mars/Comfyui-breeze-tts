"""Read desktop ``.t8voice.zip`` bundles without extracting them.

The desktop application treats a voice bundle as untrusted input.  ComfyUI
must do the same: every member name, declared size and digest is checked before
an audio payload is decoded.  This module deliberately has no dependency on
the desktop runtime so the custom node remains portable and offline.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf
import torch


BUNDLE_SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 132 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 130 * 1024 * 1024
MAX_REFERENCE_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MEMBERS = 16
MAX_REFERENCE_SECONDS = 60.0
MAX_COMPRESSION_RATIO = 500
_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_member_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("音色包包含无效文件名。")
    normalised = value.replace("\\", "/")
    path = PurePosixPath(normalised)
    if (
        path.is_absolute()
        or normalised.startswith("//")
        or re.match(r"^[A-Za-z]:", normalised)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("音色包包含路径穿越或绝对路径。")
    for part in path.parts:
        if ":" in part or part.endswith((" ", ".")):
            raise ValueError("音色包包含 Windows 不安全路径。")
    return path.as_posix()


def _safe_source(source: str | Path) -> Path:
    raw = str(source or "").strip().strip('"')
    if not raw:
        raise ValueError("bundle_path 不能为空。")
    path = Path(raw).expanduser()
    if path.is_symlink():
        raise ValueError("为防止路径替换，音色包不能是符号链接。")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("音色包不存在或路径无法访问。") from exc
    if not resolved.is_file() or not resolved.name.lower().endswith(".t8voice.zip"):
        raise ValueError("请选择桌面版导出的 .t8voice.zip 文件。")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise ValueError("音色包为空或压缩文件超过 132 MiB。")
    return resolved


def _clean_text(value: Any, *, field: str, limit: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"音色包 {field} 不能为空。")
    if len(text) > limit:
        raise ValueError(f"音色包 {field} 超过 {limit} 个字符。")
    return text


def _read_manifest_and_files(source: str | Path) -> tuple[Path, dict[str, Any], dict[str, bytes]]:
    path = _safe_source(source)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_MEMBERS:
                raise ValueError("音色包成员数量无效。")

            by_name: dict[str, zipfile.ZipInfo] = {}
            seen: set[str] = set()
            total = 0
            for info in infos:
                name = _safe_member_name(info.filename)
                folded = name.casefold()
                if folded in seen:
                    raise ValueError("音色包包含 Windows 大小写重复路径。")
                seen.add(folded)
                unix_kind = (info.external_attr >> 16) & 0o170000
                if info.is_dir() or unix_kind == 0o120000:
                    raise ValueError("音色包不能包含目录或符号链接。")
                if info.flag_bits & 0x1:
                    raise ValueError("音色包不能包含加密成员。")
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError("音色包使用了不支持的压缩算法。")
                if info.file_size < 0 or info.file_size > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("音色包成员过大。")
                if info.file_size > 1024 and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
                    raise ValueError("音色包成员压缩率异常。")
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("音色包解压后总大小超过 130 MiB。")
                by_name[name] = info

            manifest_info = by_name.get("manifest.json")
            if manifest_info is None or manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("音色包缺少有效 manifest.json。")
            try:
                manifest = json.loads(archive.read(manifest_info).decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("音色包清单不是有效 UTF-8 JSON。") from exc
            if not isinstance(manifest, dict):
                raise ValueError("音色包清单必须是 JSON 对象。")
            if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION or manifest.get("type") != "t8voice":
                raise ValueError("不支持的音色包版本或类型。")

            declared = manifest.get("files")
            if not isinstance(declared, list):
                raise ValueError("音色包 files 清单无效。")
            payloads: dict[str, bytes] = {}
            declared_names = {"manifest.json"}
            declared_folded = {"manifest.json"}
            for entry in declared:
                if not isinstance(entry, dict):
                    raise ValueError("音色包 files 清单项无效。")
                name = _safe_member_name(entry.get("path"))
                folded = name.casefold()
                if folded in declared_folded:
                    raise ValueError("音色包 files 清单包含重复路径。")
                declared_names.add(name)
                declared_folded.add(folded)
                info = by_name.get(name)
                if info is None:
                    raise ValueError("音色包缺少 files 清单声明的文件。")
                try:
                    size = int(entry.get("size"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("音色包文件大小声明无效。") from exc
                digest = str(entry.get("sha256") or "").lower()
                if size < 0 or not _SHA256.fullmatch(digest):
                    raise ValueError("音色包文件大小或 SHA-256 声明无效。")
                payload = archive.read(info)
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("音色包文件大小或 SHA-256 校验失败。")
                payloads[name] = payload
            if set(by_name) != declared_names:
                raise ValueError("音色包包含未在 files 清单中声明的文件。")
    except zipfile.BadZipFile as exc:
        raise ValueError("音色包不是有效 ZIP 文件。") from exc
    return path, manifest, payloads


def load_voice_bundle(source: str | Path) -> tuple[dict[str, Any], bytes | None, str | None]:
    """Return a validated public voice profile and its in-memory audio payload."""
    path, manifest, payloads = _read_manifest_and_files(source)
    raw_voice = manifest.get("voice")
    if not isinstance(raw_voice, dict):
        raise ValueError("音色包缺少 voice 对象。")
    mode = str(raw_voice.get("mode") or "design").strip().lower()
    if mode not in {"design", "clone", "direction"}:
        raise ValueError("音色包 voice.mode 必须是 design、clone 或 direction。")
    language = str(raw_voice.get("language") or "auto").strip().lower()
    if language not in {"auto", "zh", "en"}:
        raise ValueError("音色包 voice.language 必须是 auto、zh 或 en。")

    reference_member = str(raw_voice.get("reference_member") or "").strip()
    reference_payload: bytes | None = None
    if reference_member:
        reference_member = _safe_member_name(reference_member)
        if Path(reference_member).suffix.lower() not in _AUDIO_SUFFIXES:
            raise ValueError("音色包参考音频格式不受支持。")
        reference_payload = payloads.get(reference_member)
        if reference_payload is None:
            raise ValueError("音色包缺少 voice.reference_member 指向的文件。")
        if not reference_payload or len(reference_payload) > MAX_REFERENCE_BYTES:
            raise ValueError("音色包参考音频为空或超过 128 MiB。")
    reference_text = _clean_text(raw_voice.get("reference_text"), field="reference_text", limit=20_000)
    if mode in {"clone", "direction"} and (reference_payload is None or not reference_text):
        raise ValueError("克隆/导演音色包必须包含参考音频和准确逐字稿。")
    instruction = _clean_text(
        raw_voice.get("instruction"),
        field="instruction",
        limit=4_000,
    )
    try:
        cfg_scale = float(raw_voice.get("cfg_scale") or (1.0 if mode == "clone" else 4.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("音色包 cfg_scale 无效。") from exc
    if not 0.1 <= cfg_scale <= 10.0:
        raise ValueError("音色包 cfg_scale 必须在 0.1 到 10.0 之间。")
    voice = {
        "schema_version": int(raw_voice.get("schema_version") or 2),
        "id": _clean_text(raw_voice.get("id"), field="id", limit=128),
        "name": _clean_text(raw_voice.get("name") or "未命名音色", field="name", limit=80),
        "mode": mode,
        "language": language,
        "instruction": instruction,
        "reference_text": reference_text,
        "cfg_scale": cfg_scale,
        "reference_member": reference_member or None,
        "bundle_path": str(path),
    }
    return voice, reference_payload, reference_member or None


def decode_reference_audio(payload: bytes | None, member_name: str | None) -> dict[str, Any]:
    """Decode a verified bundle member to ComfyUI's standard AUDIO mapping."""
    if payload is None:
        return {"waveform": torch.empty((1, 1, 0), dtype=torch.float32), "sample_rate": 24_000}
    try:
        with sf.SoundFile(io.BytesIO(payload)) as audio_file:
            sample_rate = int(audio_file.samplerate)
            channels = int(audio_file.channels)
            frames = int(audio_file.frames)
            if sample_rate <= 0 or channels <= 0 or channels > 8 or frames <= 0:
                raise ValueError("参考音频的采样率、声道数或帧数无效。")
            if frames / float(sample_rate) > MAX_REFERENCE_SECONDS:
                raise ValueError("参考音频超过 60 秒上限。")
            samples = audio_file.read(dtype="float32", always_2d=True)
    except ValueError:
        raise
    except Exception as exc:
        label = member_name or "reference audio"
        raise ValueError(
            f"无法解码音色包中的 {label}；请在桌面版使用 WAV、FLAC 或宿主 libsndfile 支持的格式重新导出。"
        ) from exc
    if (
        samples.ndim != 2
        or samples.shape[0] <= 0
        or samples.shape[1] != channels
        or samples.shape[0] / float(sample_rate) > MAX_REFERENCE_SECONDS
        or not bool(np.isfinite(samples).all())
    ):
        raise ValueError("参考音频数据为空、截断或包含 NaN/Inf。")
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T)).unsqueeze(0)
    return {"waveform": waveform.contiguous(), "sample_rate": sample_rate}


def bundle_fingerprint(source: str | Path) -> str:
    """Stable ComfyUI change token; validates the path but does not parse ZIP data."""
    path = _safe_source(source)
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
