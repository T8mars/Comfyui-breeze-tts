# comfyui-breeze-tts-T8

非官方 Breeze TTS 2 ComfyUI 配套节点。v0.2.8 提供 8 个可组合节点、四份可直接拖入画布的前端工作流，并在每份工作流中展示行内声音事件语法。

## 安装

推荐在 ComfyUI-Manager 中搜索 **Breeze TTS 2 · T8star-Aix** 并安装。也可以手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-breeze-tts.git comfyui-breeze-tts-T8
python -m pip install -r comfyui-breeze-tts-T8/requirements.txt
```

依赖安装完全交给 ComfyUI-Manager 的标准管线；节点自身不会调用 `pip`。依赖清单不声明 `torch`、`torchaudio`、`torchvision`、`transformers`、`tokenizers` 或 `numpy`，这些核心包沿用宿主环境，并在节点加载时执行兼容性检查。

如果是已经能正常运行的 ComfyUI Portable，并且希望手动安装时绝对禁止依赖解析器触碰宿主核心包，可在 ComfyUI 根目录执行：

```powershell
.\python_embeded\python.exe -s -m pip install --no-deps -r .\custom_nodes\comfyui-breeze-tts-T8\requirements.txt
```

`--no-deps` 只安装清单中直接列出的轻量包，不解析它们的传递依赖。若随后提示缺少轻量依赖，优先改用 ComfyUI-Manager 修复；不要单独执行 `pip install -U torch transformers tokenizers numpy`。

首次执行模型加载器时可自动下载官方 `BreezeBlue/Breeze-TTS-2` 固定 revision。也可以手动放到：

`ComfyUI/models/breeze_tts/BreezeBlue_Breeze-TTS-2`

加载前会检查模型配置、Tokenizer、Codec、权重索引及索引引用的每个分片。中断的下载可在
`download_if_missing=true` 时继续；检查失败会显示具体缺失/损坏文件和修复路径。

## 兼容性

- Python：`>=3.10,<3.13`
- Transformers：`>=4.57,<6`
- 已验证目标：Transformers 4.57.3 与 5.16.1；ComfyUI 0.34.0 / 前端 1.51.9 的画布工作流
- Torch：沿用 ComfyUI 自己的版本；GPU 推理建议支持 BF16

节点内置了 T5Gemma2 与 Qwen3 TTS codec 的跨版本兼容实现。4.57.x 使用随节点提供的 T5Gemma2 兼容层，5.x 优先使用 Transformers 自带实现；因版本变化产生的 causal-mask、StaticCache 和 RoPE 接口差异在节点内部适配。版本不在支持范围时会在节点注册阶段直接报告原因，不会静默覆盖宿主环境。

兼容检查只通过包元数据读取 `torch / torchaudio / torchvision / transformers / tokenizers / numpy` 的宿主版本，不会导入安装器、修改 `sys.path`、写入 `site-packages` 或执行子进程。节点也没有 `install.py` 和运行时安装逻辑。

## 节点与输入输出契约

| 节点 | 主要输入 | 输出 |
| --- | --- | --- |
| `T8 模型加载器` | 精度、设备、注意力、许可证确认 | `BREEZE_T8_MODEL`、模型信息 JSON |
| `T8 声音设计` | 台词、声音描述、CFG | `BREEZE_T8_REQUEST` |
| `T8 声音克隆` | 台词、标准 `AUDIO`、准确逐字稿 | `BREEZE_T8_REQUEST` |
| `T8 声音导演` | 克隆输入、逐句自然语言指令 | `BREEZE_T8_REQUEST` |
| `T8 桌面音色包` | `.t8voice.zip`、台词、逐句指令 | Request、标准 `AUDIO`、音色信息 JSON |
| `T8 逐句情感` | 上游 Request、继承/覆盖/中性 | 新的 Request（不修改上游） |
| `T8 生成设置` | 采样参数与 Seed | `BREEZE_T8_SETTINGS` |
| `T8 生成音频` | Model、Request、Settings | 标准 ComfyUI `AUDIO`、生成信息 JSON |

Request 和 Settings 会在模型恢复到 GPU 之前验证。空台词、空逐字稿、非法 AUDIO 形状、非正温度、非正重复惩罚或缺失设置字段会直接给出字段级错误，不会等到模型推理阶段才失败。输出 `AUDIO` 固定遵循 `{"waveform": [batch, channels, samples], "sample_rate": int}`。

## 使用顺序

1. `T8 模型加载器`
2. `T8 声音设计`、`T8 声音克隆` 或 `T8 声音导演`
3. `T8 生成设置`
4. `T8 生成音频`

## 工作流示例

`examples/` 同时提供两类 JSON，格式不同，不能混用：

- `*_workflow.json`：**ComfyUI 前端工作流**。直接把文件拖进 ComfyUI 画布，或使用“工作流 → 打开”导入。
- `*_api.json`：仅供 `/prompt` HTTP API 或脚本调用，不能拖入画布。

可直接导入的四份前端工作流：

- `voice_design_workflow.json`：声音设计；
- `voice_clone_workflow.json`：声音克隆；
- `voice_direction_workflow.json`：参考音色加情绪/语速导演；
- `voice_bundle_workflow.json`：读取桌面版导出的 `.t8voice.zip` 音色包。

导入后先在粉色分组中的 `① 模型加载器` 阅读许可证并勾选 `accept_model_license`。Clone/Direction 工作流还必须在 `LoadAudio` 中重新选择自己的参考音频，并填写准确逐字稿；示例不会内嵌 `reference.wav`。因此首次打开这两份工作流时，ComfyUI 会提示“缺少媒体输入 reference.wav”，这是等待用户选择参考音频的正常提示，不是节点缺失。关闭提示，在 `LoadAudio` 中点击“选择文件上传”即可。Voice Bundle 工作流需把示例路径改成真实的 `.t8voice.zip` 绝对路径。

如果导入后节点显示红色“缺失”，请确认节点目录是 `ComfyUI/custom_nodes/comfyui-breeze-tts-T8`，重启 ComfyUI 后再导入 `*_workflow.json`，不要导入同名的 `*_api.json`。

## 行内声音事件

声音事件直接写入任意 Design、Clone、Direction 或 Voice Bundle 节点的 `text` 台词框，不需要额外节点。四份 UI 工作流底部均有粉色语法提示区，各请求节点的悬浮说明也会显示相同提示。

- 英文：`(laugh)`、`(cough)`、`(clears throat)`、`(sigh)`
- 中文：`[笑]`、`[咳嗽]`、`[清嗓子]`、`[叹气]`

例如：`[清嗓子] 接下来宣布今天的安排。` 或 `(sigh) I thought we had more time.`。这些标记会原样交给 Breeze TTS 2；它们是生成提示，实际强度仍会受台词、指令、采样参数和 Seed 影响。

## 桌面音色包与逐句情感

桌面版音色库可导出 `*.t8voice.zip`。在 `T8 桌面音色包` 节点中填写该文件的本地绝对路径和本句文本，输出的 `request` 可直接连接 `T8 生成音频`，`reference_audio` 是标准 ComfyUI `AUDIO`，可用于预览或其他音频节点。

逐句表达使用 Breeze 原生的自然语言 direction，而不是伪造 IndexTTS 的 8 维情感向量：

- `inherit`：继承音色包的声音描述或导演指令；
- `override`：本句指令替换默认指令；克隆音色会自动转为 Direction 请求，同时保留同一参考音色；
- `neutral`：使用清晰自然的中性指令；参考音色仍保留说话人身份。

也可以把任意 Design/Clone/Direction 请求接入独立的 `T8 逐句情感` 节点，按台词逐句覆盖。节点复制 request 后再修改，不会污染上游节点或其他分支。

音色包按不可信文件处理：只离线读取、不联网、不向磁盘解压；拒绝绝对路径、路径穿越、符号链接、加密成员、Windows 大小写重复路径、未声明成员和异常压缩率；限制成员数、压缩/解压大小和参考音频时长，并逐项校验声明大小与 SHA-256。参考音频由内存直接解码为 `[batch, channels, samples]` 的 `float32` AUDIO。为获得最稳定的跨平台解码效果，建议桌面端导出 WAV 或 FLAC。

声音克隆与声音导演必须提供参考音频的准确逐字稿。模型仅限其许可证允许的研究、教育与非商业用途，详见 `MODEL_LICENSE`。
参考音频最长 60 秒，节点会在上传到 GPU 和 Codec 编码前检查原始波形时长。

## 来源与声明

本节点不是 BreezeBlue 官方产品。模型与核心项目来自 `breezeblue-ai/breeze-tts`；兼容推理路径部分改编自 `Saganaki22/ComfyUI-Breeze-TTS-2`。完整归属见 `THIRD_PARTY_NOTICES.md`。

## 发布信息

- GitHub：<https://github.com/T8mars/Comfyui-breeze-tts>
- Comfy Registry Publisher：`t8star`
- Registry 节点 ID：`comfyui-breeze-tts-T8`
- 当前版本：`0.2.8`
