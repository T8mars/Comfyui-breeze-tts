"""Composable ComfyUI nodes for the unofficial T8 Breeze TTS 2 integration.

The inference path is adapted from Saganaki22/ComfyUI-Breeze-TTS-2 and the
official breezeblue-ai/breeze-tts implementation, both Apache-2.0.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import secrets
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from . import loader, native, runtime

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
        return ({"mode": "design", "text": text, "instruction": voice_description, "cfg_scale": cfg_scale},)


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
        return ({
            "mode": "clone",
            "text": text,
            "reference_audio": reference_audio,
            "reference_text": reference_text,
            "instruction": instruction,
            "cfg_scale": cfg_scale,
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
        return ({
            "mode": "direction",
            "text": text,
            "reference_audio": reference_audio,
            "reference_text": reference_text,
            "instruction": direction,
            "cfg_scale": cfg_scale,
        },)


class BreezeT8GenerationSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_new_tokens": ("INT", {"default": 1500, "min": 64, "max": 3000, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "depth_temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05}),
                "depth_top_k": ("INT", {"default": 50, "min": 0, "max": 1024}),
                "depth_top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "集中管理可复现的采样参数。"

    def build(self, **kwargs):
        return (dict(kwargs),)


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
    "T8_BreezeTTS_GenerationSettings": BreezeT8GenerationSettings,
    "T8_BreezeTTS_Generate": BreezeT8Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T8_BreezeTTS_ModelLoader": "Breeze TTS 2 · T8 模型加载器",
    "T8_BreezeTTS_DesignRequest": "Breeze TTS 2 · T8 声音设计",
    "T8_BreezeTTS_CloneRequest": "Breeze TTS 2 · T8 声音克隆",
    "T8_BreezeTTS_DirectionRequest": "Breeze TTS 2 · T8 声音导演",
    "T8_BreezeTTS_GenerationSettings": "Breeze TTS 2 · T8 生成设置",
    "T8_BreezeTTS_Generate": "Breeze TTS 2 · T8 生成音频",
}
