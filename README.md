# comfyui-breeze-tts-T8

非官方 Breeze TTS 2 ComfyUI 配套节点。节点包提供 6 个可组合节点：模型加载、声音设计、声音克隆、声音导演、生成设置和音频生成。

## 安装

推荐在 ComfyUI-Manager 中搜索 **Breeze TTS 2 · T8star-Aix** 并安装。也可以手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-breeze-tts.git
python -m pip install -r Comfyui-breeze-tts/requirements.txt
```

依赖安装完全交给 ComfyUI-Manager 的标准管线；节点自身不会调用 `pip`。依赖清单不声明 `torch`、`torchaudio`、`torchvision`、`transformers`、`tokenizers` 或 `numpy`，这些核心包沿用宿主环境，并在节点加载时执行兼容性检查。

首次执行模型加载器时可自动下载官方 `BreezeBlue/Breeze-TTS-2` 固定 revision。也可以手动放到：

`ComfyUI/models/breeze_tts/BreezeBlue_Breeze-TTS-2`

## 兼容性

- Python：3.10+
- Transformers：`>=4.57,<6`
- 已验证目标：Transformers 4.57.3 与 5.16.1；ComfyUI 0.33.0 API 工作流
- Torch：沿用 ComfyUI 自己的版本；GPU 推理建议支持 BF16

节点内置了 T5Gemma2 与 Qwen3 TTS codec 的跨版本兼容实现。版本不在支持范围时会在节点注册阶段直接报告原因，不会静默覆盖宿主环境。

## 使用顺序

1. `T8 模型加载器`
2. `T8 声音设计`、`T8 声音克隆` 或 `T8 声音导演`
3. `T8 生成设置`
4. `T8 生成音频`

`examples/` 提供 Design、Clone、Direction 三份 API 工作流。导入前请把模型加载器的 `accept_model_license` 改为 `true`；Clone/Direction 还需替换 `reference.wav` 与准确逐字稿。

声音克隆与声音导演必须提供参考音频的准确逐字稿。模型仅限其许可证允许的研究、教育与非商业用途，详见 `MODEL_LICENSE`。

## 来源与声明

本节点不是 BreezeBlue 官方产品。模型与核心项目来自 `breezeblue-ai/breeze-tts`；兼容推理路径部分改编自 `Saganaki22/ComfyUI-Breeze-TTS-2`。完整归属见 `THIRD_PARTY_NOTICES.md`。

## 发布信息

- GitHub：<https://github.com/T8mars/Comfyui-breeze-tts>
- Comfy Registry Publisher：`t8star`
- Registry 节点 ID：`comfyui-breeze-tts-T8`
- 当前版本：`0.1.0`
