"""Composable ComfyUI nodes for the unofficial T8 Breeze TTS 2 integration.

The inference path is adapted from Saganaki22/ComfyUI-Breeze-TTS-2 and the
official breezeblue-ai/breeze-tts implementation, both Apache-2.0.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import secrets
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from . import compat, loader, native, runtime, voice_bundle

CATEGORY = "T8star-Aix/Audio/Breeze TTS"
MODEL_TYPE = "BREEZE_T8_MODEL"
REQUEST_TYPE = "BREEZE_T8_REQUEST"
SETTINGS_TYPE = "BREEZE_T8_SETTINGS"
_GENERATION_LOCK = loader.GENERATION_LOCK
_REFERENCE_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_REFERENCE_CACHE_LIMIT = 8

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


def _text(default: str, tooltip: str) -> tuple:
    return ("STRING", {"default": default, "multiline": True, "tooltip": tooltip})


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} 不能为空。")
    return text


def _finite_float(value: Any, field_name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数字。") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{field_name} 必须在 {minimum:g} 到 {maximum:g} 之间。")
    return number


def _bounded_int(value: Any, field_name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是整数。")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是整数。") from exc
    if number != value or not minimum <= number <= maximum:
        raise ValueError(f"{field_name} 必须在 {minimum} 到 {maximum} 之间。")
    return number


def _validate_audio_contract(audio: Any, field_name: str = "reference_audio") -> None:
    if not isinstance(audio, dict):
        raise ValueError(f"{field_name} 必须是 ComfyUI AUDIO。")
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.numel() <= 0:
        raise ValueError(f"{field_name}.waveform 必须是非空的 [batch, channels, samples] Tensor。")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{field_name}.waveform 不能包含 NaN 或 Inf。")
    try:
        sample_rate = int(audio.get("sample_rate", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}.sample_rate 必须是正整数。") from exc
    if sample_rate <= 0:
        raise ValueError(f"{field_name}.sample_rate 必须是正整数。")


def _validate_request_contract(request: Any) -> None:
    if not isinstance(request, dict):
        raise ValueError("request 必须来自 T8 Breeze 请求节点。")
    mode = str(request.get("mode") or "").strip().lower()
    if mode not in {"design", "clone", "direction"}:
        raise ValueError("request.mode 必须是 design、clone 或 direction。")
    _required_text(request.get("text"), "text")
    _required_text(request.get("instruction"), "instruction")
    _finite_float(request.get("cfg_scale", 1.0), "cfg_scale", 0.1, 10.0)
    if mode in {"clone", "direction"}:
        _validate_audio_contract(request.get("reference_audio"))
        _required_text(request.get("reference_text"), "reference_text")


def _validate_settings_contract(settings: Any) -> None:
    if not isinstance(settings, dict):
        raise ValueError("settings 必须来自 T8 生成设置节点。")
    required = {
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "depth_temperature",
        "depth_top_k",
        "depth_top_p",
        "seed",
    }
    missing = sorted(required.difference(settings))
    if missing:
        raise ValueError("settings 缺少字段: " + ", ".join(missing))
    _bounded_int(settings["max_new_tokens"], "max_new_tokens", 64, 3000)
    _bounded_int(settings["top_k"], "top_k", 0, 1024)
    _bounded_int(settings["depth_top_k"], "depth_top_k", 0, 1024)
    _bounded_int(settings["seed"], "seed", 0, 2**31 - 1)
    _finite_float(settings["temperature"], "temperature", 0.01, 2.0)
    _finite_float(settings["depth_temperature"], "depth_temperature", 0.01, 2.0)
    _finite_float(settings["repetition_penalty"], "repetition_penalty", 0.01, 2.0)
    _finite_float(settings["top_p"], "top_p", 0.01, 1.0)
    _finite_float(settings["depth_top_p"], "depth_top_p", 0.01, 1.0)


class BreezeT8ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (loader.DTYPE_OPTIONS, {"default": "auto"}),
                "device": (loader.DEVICE_OPTIONS, {"default": "auto"}),
                "attention": (loader.ATTENTION_OPTIONS, {"default": "auto"}),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "缺少时从 BreezeBlue/Breeze-TTS-2 的固定 revision 下载官方模型。",
                    },
                ),
                "accept_model_license": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "I accept",
                        "label_off": "Not accepted",
                        "tooltip": "确认接受 MODEL_LICENSE 的研究与非商业条款后才可下载或加载模型。",
                    },
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "加载官方 Breeze TTS 2 分片权重；不覆盖 ComfyUI 的 Torch/Transformers。"

    def load(self, dtype, device, attention, download_if_missing, accept_model_license):
        if not bool(accept_model_license):
            raise RuntimeError("请先阅读节点目录中的 MODEL_LICENSE，并勾选 accept_model_license。")
        try:
            bundle = loader.load_breeze_bundle(
                loader.BF16_LABEL,
                dtype,
                device,
                attention,
                bool(download_if_missing),
                "eager",
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "Breeze TTS 2 模型加载时显存不足。请停止其他工作流、卸载占用显存的模型，"
                "或把 device 改为 CPU 后重试；无需重新安装 Torch/Transformers。"
            ) from exc
        except Exception as exc:
            # Integrity/download failures already contain exact file names;
            # this suffix also makes native/tokenizer load failures actionable.
            raise RuntimeError(
                f"Breeze TTS 2 模型加载失败: {type(exc).__name__}: {exc}。"
                "修复顺序：保持 download_if_missing=true 再执行以续传；"
                "若提示某个文件损坏，关闭 ComfyUI 后仅删除该文件再重试；"
                "不要用本节点覆盖 ComfyUI 的 Torch、Transformers、Tokenizers 或 NumPy。"
            ) from exc
        info = {
            "model_dir": str(bundle.model_dir),
            "revision": loader.MODEL_REVISION,
            "device": str(bundle.device),
            "dtype": bundle.dtype_name,
            "attention": bundle.attention,
            "compatibility": compat.check_transformers(raise_on_error=False).to_dict(),
        }
        return bundle, json.dumps(info, ensure_ascii=False, indent=2)


class BreezeT8DesignRequest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": _text("欢迎使用 Breeze TTS 2。", "要合成的文本。"),
                "voice_description": _text(
                    "一位温柔自信的年轻女性，声音清晰，语气亲切。",
                    "无参考音频的声音描述；建议与正文使用同一语言。",
                ),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = (REQUEST_TYPE,)
    RETURN_NAMES = ("request",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "创建零样本声音设计请求。"

    def build(self, text, voice_description, cfg_scale):
        return ({
            "mode": "design",
            "text": _required_text(text, "text"),
            "instruction": _required_text(voice_description, "voice_description"),
            "cfg_scale": _finite_float(cfg_scale, "cfg_scale", 0.1, 10.0),
        },)


class BreezeT8CloneRequest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": _text("很高兴再次听到你的声音。", "要合成的文本。"),
                "reference_audio": ("AUDIO",),
                "reference_text": _text("参考音频的准确逐字稿。", "必须与参考音频准确对应。"),
            },
            "optional": {
                "instruction": _text(runtime.DEFAULT_INSTRUCTION, "可选的自然语言表演指令。"),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = (REQUEST_TYPE,)
    RETURN_NAMES = ("request",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "从参考音频与准确逐字稿创建声音克隆请求。"

    def build(self, text, reference_audio, reference_text, instruction=runtime.DEFAULT_INSTRUCTION, cfg_scale=1.0):
        _validate_audio_contract(reference_audio)
        return ({
            "mode": "clone",
            "text": _required_text(text, "text"),
            "reference_audio": reference_audio,
            "reference_text": _required_text(reference_text, "reference_text"),
            "instruction": _required_text(instruction, "instruction"),
            "cfg_scale": _finite_float(cfg_scale, "cfg_scale", 0.1, 10.0),
        },)


class BreezeT8DirectionRequest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": _text("我们需要认真讨论一下昨晚发生的事情。", "要合成的文本。"),
                "reference_audio": ("AUDIO",),
                "reference_text": _text("参考音频的准确逐字稿。", "必须与参考音频准确对应。"),
                "direction": _text("语速放慢，语气克制而严肃。", "音色不变时的情绪、节奏和表达指令。"),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = (REQUEST_TYPE,)
    RETURN_NAMES = ("request",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "保留参考说话人身份，同时控制情绪、语速与表达。"

    def build(self, text, reference_audio, reference_text, direction, cfg_scale):
        _validate_audio_contract(reference_audio)
        return ({
            "mode": "direction",
            "text": _required_text(text, "text"),
            "reference_audio": reference_audio,
            "reference_text": _required_text(reference_text, "reference_text"),
            "instruction": _required_text(direction, "direction"),
            "cfg_scale": _finite_float(cfg_scale, "cfg_scale", 0.1, 10.0),
        },)


class BreezeT8VoiceBundleRequest:
    """Bridge a desktop voice-library bundle into a standard Breeze request."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle_path": (
                    "STRING",
                    {
                        "default": "voice.t8voice.zip",
                        "tooltip": "桌面版音色库导出的本地 .t8voice.zip；节点只离线读取，不会联网或解压到磁盘。",
                    },
                ),
                "text": _text("欢迎使用桌面版与 ComfyUI 共用的音色。", "本次要合成的台词。"),
                "line_direction_mode": (
                    ["inherit", "override", "neutral"],
                    {
                        "default": "inherit",
                        "tooltip": "inherit 继承音色；override 使用本句指令；neutral 使用自然、清晰的中性表达。",
                    },
                ),
            },
            "optional": {
                "line_direction": _text("", "逐句自然语言情感/语速/表达指令；override 时必填。"),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "0 表示沿用音色包默认值；其他值覆盖本句 CFG。",
                    },
                ),
            },
        }

    RETURN_TYPES = (REQUEST_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("request", "reference_audio", "voice_info")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "安全读取桌面版 .t8voice.zip，并生成可直接连接 T8 生成节点的请求。"

    @classmethod
    def IS_CHANGED(cls, bundle_path, **_kwargs):
        try:
            return voice_bundle.bundle_fingerprint(bundle_path)
        except ValueError as exc:
            return f"invalid:{exc}"

    def build(self, bundle_path, text, line_direction_mode, line_direction="", cfg_scale=0.0):
        text = str(text or "").strip()
        if not text:
            raise ValueError("text 不能为空。")
        profile, payload, member = voice_bundle.load_voice_bundle(bundle_path)
        reference_audio = voice_bundle.decode_reference_audio(payload, member)

        direction_mode = str(line_direction_mode or "inherit").strip().lower()
        if direction_mode not in {"inherit", "override", "neutral"}:
            raise ValueError("line_direction_mode 必须是 inherit、override 或 neutral。")
        profile_mode = profile["mode"]
        has_reference = payload is not None
        instruction = str(profile.get("instruction") or runtime.DEFAULT_INSTRUCTION).strip()
        request_mode = profile_mode
        if direction_mode == "override":
            instruction = str(line_direction or "").strip()
            if not instruction:
                raise ValueError("line_direction_mode=override 时 line_direction 不能为空。")
            # A clone with per-line direction becomes a Direction request while
            # retaining the same verified speaker reference.
            request_mode = "direction" if has_reference else "design"
        elif direction_mode == "neutral":
            instruction = runtime.DEFAULT_INSTRUCTION
            request_mode = "clone" if has_reference else "design"

        selected_cfg = float(cfg_scale)
        if selected_cfg == 0.0:
            selected_cfg = float(profile["cfg_scale"])
        if not 0.1 <= selected_cfg <= 10.0:
            raise ValueError("cfg_scale 必须为 0（继承）或 0.1 到 10.0。")
        request = {
            "mode": request_mode,
            "text": text,
            "instruction": instruction,
            "cfg_scale": selected_cfg,
            "language": profile["language"],
            "voice_id": profile["id"],
            "voice_name": profile["name"],
            "line_direction_mode": direction_mode,
        }
        if has_reference:
            request["reference_audio"] = reference_audio
            request["reference_text"] = profile["reference_text"]
        public_info = {
            key: value
            for key, value in profile.items()
            if key not in {"bundle_path"}
        }
        public_info.update({"effective_mode": request_mode, "line_direction_mode": direction_mode})
        return request, reference_audio, json.dumps(public_info, ensure_ascii=False, indent=2)


class BreezeT8LineDirection:
    """Apply a non-destructive per-line direction override to any request."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request": (REQUEST_TYPE,),
                "direction_mode": (["inherit", "override", "neutral"], {"default": "inherit"}),
                "direction": _text("语气温和而坚定，停顿自然。", "override 模式使用的本句指令。"),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "0 保留上游设置；其他值仅覆盖本句 CFG。",
                    },
                ),
            }
        }

    RETURN_TYPES = (REQUEST_TYPE,)
    RETURN_NAMES = ("request",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "为单句设置继承、覆盖或中性表达；不会修改上游 request。"

    def apply(self, request, direction_mode, direction, cfg_scale):
        if not isinstance(request, dict):
            raise ValueError("request 必须来自 T8 Breeze 请求节点。")
        result = dict(request)
        mode = str(direction_mode or "inherit").strip().lower()
        if mode not in {"inherit", "override", "neutral"}:
            raise ValueError("direction_mode 必须是 inherit、override 或 neutral。")
        has_reference = result.get("reference_audio") is not None
        if mode == "override":
            instruction = str(direction or "").strip()
            if not instruction:
                raise ValueError("direction_mode=override 时 direction 不能为空。")
            result["instruction"] = instruction
            result["mode"] = "direction" if has_reference else "design"
        elif mode == "neutral":
            result["instruction"] = runtime.DEFAULT_INSTRUCTION
            result["mode"] = "clone" if has_reference else "design"
        selected_cfg = float(cfg_scale)
        if selected_cfg != 0.0:
            if not 0.1 <= selected_cfg <= 10.0:
                raise ValueError("cfg_scale 必须为 0（保留）或 0.1 到 10.0。")
            result["cfg_scale"] = selected_cfg
        result["line_direction_mode"] = mode
        return (result,)


class BreezeT8GenerationSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_new_tokens": ("INT", {"default": 1500, "min": 64, "max": 3000, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.01, "max": 2.0, "step": 0.05}),
                "depth_temperature": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 2.0, "step": 0.05}),
                "depth_top_k": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "depth_top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "集中管理可复现的采样参数。"

    def build(self, **kwargs):
        settings = dict(kwargs)
        _validate_settings_contract(settings)
        return (settings,)


@contextlib.contextmanager
def _isolated_rng(bundle, requested_seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = []
    if bundle.device.type == "cuda":
        devices = [bundle.device.index if bundle.device.index is not None else torch.cuda.current_device()]
    actual_seed = int(requested_seed) if int(requested_seed) > 0 else secrets.randbelow(2**31 - 1) + 1
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            runtime.set_all_seeds(actual_seed)
            yield actual_seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _reference_cache_key(bundle, wav: torch.Tensor, sample_rate: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(bundle.model_dir).encode("utf-8"))
    digest.update(int(sample_rate).to_bytes(4, "little", signed=False))
    digest.update(wav.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _generate_audio(bundle, request: dict[str, Any], settings: dict[str, Any]) -> tuple[dict, dict]:
    started = time.perf_counter()
    _validate_request_contract(request)
    _validate_settings_contract(settings)
    text = str(request.get("text", "")).strip()
    if not text:
        raise ValueError("text 不能为空。")
    instruction = str(request.get("instruction") or runtime.DEFAULT_INSTRUCTION).strip()
    ref_audio = request.get("reference_audio")
    ref_text = str(request.get("reference_text") or "").strip() or None
    cfg_scale = float(request.get("cfg_scale", 1.0))

    loader.resume_bundle_to_device(bundle)
    ref_codes = None
    reference_cache_hit = False
    if ref_audio is not None:
        if not ref_text:
            raise ValueError("参考音频必须提供准确逐字稿 reference_text。")
        wav, sample_rate = runtime.comfy_audio_to_tensor(ref_audio)
        if wav.numel() == 0:
            raise ValueError("参考音频为空。")
        if sample_rate <= 0:
            raise ValueError(f"参考音频采样率无效: {sample_rate}。")
        # Check the original CPU waveform before hashing, resampling or codec
        # encoding.  A long reference must not reach the GPU and OOM first.
        seconds = wav.numel() / float(sample_rate)
        if seconds > runtime.MAX_REFERENCE_SECONDS:
            raise ValueError(f"参考音频约 {seconds:.1f} 秒，超过 {runtime.MAX_REFERENCE_SECONDS:.0f} 秒上限。")
        cache_key = _reference_cache_key(bundle, wav, sample_rate)
        ref_codes = _REFERENCE_CACHE.get(cache_key)
        if ref_codes is not None:
            reference_cache_hit = True
            _REFERENCE_CACHE.move_to_end(cache_key)
        else:
            ref_codes = runtime.encode_reference_audio(bundle.codec, wav, sample_rate)
            _REFERENCE_CACHE[cache_key] = ref_codes
            _REFERENCE_CACHE.move_to_end(cache_key)
            while len(_REFERENCE_CACHE) > _REFERENCE_CACHE_LIMIT:
                _REFERENCE_CACHE.popitem(last=False)

    if ref_codes is None:
        cond = runtime.design_segments(text, instruction)
        negative = runtime.design_negative_segments(text)
    else:
        cond = runtime.ref_segments(ref_text, text, instruction)
        negative = runtime.ref_segments(ref_text, text, instruction, with_instruction=False)

    with _isolated_rng(bundle, int(settings["seed"])) as actual_seed:
        embeds, mask, positions, prefill_len = runtime.build_generation_batch(
            bundle.model,
            bundle.tokenizer,
            cond_segments=cond,
            negative_segments=negative if cfg_scale != 1.0 else None,
            ref_codes=ref_codes,
            cfg_scale=cfg_scale,
            device=bundle.device,
        )
        max_frames = min(int(settings["max_new_tokens"]), runtime.MAX_SEQ_LEN - 1 - prefill_len)
        if max_frames < 64:
            raise ValueError("提示内容或参考音频过长，剩余音频帧不足。")

        params = runtime.GenerationParams(
            max_new_tokens=max_frames,
            temperature=float(settings["temperature"]),
            top_k=int(settings["top_k"]),
            top_p=float(settings["top_p"]),
            repetition_penalty=float(settings["repetition_penalty"]),
            depth_temperature=float(settings["depth_temperature"]),
            depth_top_k=int(settings["depth_top_k"]),
            depth_top_p=float(settings["depth_top_p"]),
        )
        pbar = ProgressBar(max_frames) if ProgressBar is not None else None

        def on_progress(current: int) -> None:
            if pbar is not None:
                pbar.update_absolute(min(current, max_frames), max_frames)
            try:
                import comfy.model_management as mm
                mm.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

        with torch.inference_mode(), native.attention_runtime(bundle.attention):
            codes = runtime.generate_codes(
                bundle.model,
                inputs_embeds=embeds,
                attention_mask=mask,
                base_positions=positions,
                prefill_len=prefill_len,
                cfg_scale=cfg_scale,
                params=params,
                progress_callback=on_progress,
                decode_mode=bundle.decode_mode,
            )
            wav = runtime.decode_codes(bundle.codec, codes)
    if wav.numel() == 0 or not bool(torch.isfinite(wav).all()):
        raise RuntimeError("模型没有生成有效音频。")
    audio = runtime.tensor_audio_to_comfy(wav)
    elapsed = time.perf_counter() - started
    duration = audio["waveform"].numel() / float(audio["sample_rate"])
    return audio, {
        "actual_seed": actual_seed,
        "elapsed_seconds": elapsed,
        "duration_seconds": duration,
        "rtf": elapsed / duration if duration > 0 else None,
        "reference_cache_hit": reference_cache_hit,
        "reference_cache_entries": len(_REFERENCE_CACHE),
    }


class BreezeT8Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": (MODEL_TYPE,), "request": (REQUEST_TYPE,), "settings": (SETTINGS_TYPE,)}}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "执行 Breeze TTS 2 推理并返回标准 ComfyUI AUDIO。"

    def generate(self, model, request, settings):
        if not loader.try_begin_generation():
            raise RuntimeError("Breeze TTS 2 正在生成；为保护缓存与显存，T8 节点会串行执行。")
        try:
            audio, metrics = _generate_audio(model, request, settings)
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError("Breeze TTS 2 显存不足。已清理临时缓存；可降低其他模型显存占用后重试。") from exc
        finally:
            loader.end_generation()
        info = {
            "mode": request.get("mode"),
            "sample_rate": int(audio["sample_rate"]),
            "seed": int(settings["seed"]),
            "model_revision": loader.MODEL_REVISION,
            **metrics,
        }
        return audio, json.dumps(info, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "T8_BreezeTTS_ModelLoader": BreezeT8ModelLoader,
    "T8_BreezeTTS_DesignRequest": BreezeT8DesignRequest,
    "T8_BreezeTTS_CloneRequest": BreezeT8CloneRequest,
    "T8_BreezeTTS_DirectionRequest": BreezeT8DirectionRequest,
    "T8_BreezeTTS_VoiceBundleRequest": BreezeT8VoiceBundleRequest,
    "T8_BreezeTTS_LineDirection": BreezeT8LineDirection,
    "T8_BreezeTTS_GenerationSettings": BreezeT8GenerationSettings,
    "T8_BreezeTTS_Generate": BreezeT8Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T8_BreezeTTS_ModelLoader": "Breeze TTS 2 · T8 模型加载器",
    "T8_BreezeTTS_DesignRequest": "Breeze TTS 2 · T8 声音设计",
    "T8_BreezeTTS_CloneRequest": "Breeze TTS 2 · T8 声音克隆",
    "T8_BreezeTTS_DirectionRequest": "Breeze TTS 2 · T8 声音导演",
    "T8_BreezeTTS_VoiceBundleRequest": "Breeze TTS 2 · T8 桌面音色包",
    "T8_BreezeTTS_LineDirection": "Breeze TTS 2 · T8 逐句情感",
    "T8_BreezeTTS_GenerationSettings": "Breeze TTS 2 · T8 生成设置",
    "T8_BreezeTTS_Generate": "Breeze TTS 2 · T8 生成音频",
}
