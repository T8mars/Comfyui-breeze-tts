"""Generate frontend-loadable ComfyUI workflow examples.

The ``*_api.json`` examples in this project are prompt objects for the HTTP
API.  ComfyUI's canvas needs the separate LiteGraph workflow format generated
here (top-level ``nodes`` and ``links`` arrays).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID


PACKAGE_VERSION = "0.2.6"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
INLINE_EVENT_GUIDE = (
    "行内声音事件｜中文：[笑] [咳嗽] [清嗓子] [叹气]｜"
    "English: (laugh) (cough) (clears throat) (sigh)"
)
WORKFLOW_IDS = {
    "voice_design_workflow.json": "533675ed-0775-4ccd-a730-49b085a2b910",
    "voice_clone_workflow.json": "2ae14050-5972-43bc-b42c-cae09ca29831",
    "voice_direction_workflow.json": "8aa43f3a-f29e-43ae-a401-83813def9bcb",
    "voice_bundle_workflow.json": "6f89a2bc-50f8-4406-aa82-2e713f6d5e64",
}


def _properties(node_type: str, *, core: bool = False) -> dict[str, str]:
    properties = {"Node name for S&R": node_type}
    properties["cnr_id"] = "comfy-core" if core else "comfyui-breeze-tts-T8"
    properties["ver"] = "0.3.60" if core else PACKAGE_VERSION
    return properties


def _node(
    node_id: int,
    node_type: str,
    pos: list[int],
    size: list[int],
    order: int,
    *,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    widgets: list[Any] | None = None,
    title: str | None = None,
    core: bool = False,
    color: str | None = None,
    bgcolor: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": _properties(node_type, core=core),
        "widgets_values": widgets or [],
    }
    if title:
        result["title"] = title
    if color:
        result["color"] = color
    if bgcolor:
        result["bgcolor"] = bgcolor
    return result


def _input(name: str, data_type: str, link: int) -> dict[str, Any]:
    return {"name": name, "type": data_type, "link": link}


def _output(name: str, data_type: str, links: list[int] | None, slot: int) -> dict[str, Any]:
    return {"name": name, "type": data_type, "links": links, "slot_index": slot}


def _loader(link: int) -> dict[str, Any]:
    return _node(
        1,
        "T8_BreezeTTS_ModelLoader",
        [0, 80],
        [340, 210],
        0,
        outputs=[
            _output("model", "BREEZE_T8_MODEL", [link], 0),
            _output("model_info", "STRING", None, 1),
        ],
        widgets=["auto", "auto", "auto", True, False],
        title="① 模型加载器 · 先勾选许可证",
        color="#5b3153",
        bgcolor="#2b1b2a",
    )


def _settings(node_id: int, link: int, order: int = 2, step: int = 3) -> dict[str, Any]:
    return _node(
        node_id,
        "T8_BreezeTTS_GenerationSettings",
        [0, 350],
        [370, 360],
        order,
        outputs=[_output("settings", "BREEZE_T8_SETTINGS", [link], 0)],
        widgets=[1500, 0.9, 50, 1.0, 1.1, 0.9, 50, 1.0, 42],
        title=f"{step} 生成设置",
        color="#4b4777",
        bgcolor="#262442",
    )


def _generate(
    node_id: int,
    model_link: int,
    request_link: int,
    settings_link: int,
    audio_links: list[int],
    order: int,
    step: int = 4,
) -> dict[str, Any]:
    return _node(
        node_id,
        "T8_BreezeTTS_Generate",
        [840, 190],
        [330, 160],
        order,
        inputs=[
            _input("model", "BREEZE_T8_MODEL", model_link),
            _input("request", "BREEZE_T8_REQUEST", request_link),
            _input("settings", "BREEZE_T8_SETTINGS", settings_link),
        ],
        outputs=[
            _output("audio", "AUDIO", audio_links, 0),
            _output("generation_info", "STRING", None, 1),
        ],
        title=f"{step} 生成音频",
        color="#7a405f",
        bgcolor="#38202f",
    )


def _audio_outputs(
    generate_id: int,
    save_link: int,
    preview_link: int,
    prefix: str,
    order: int,
    step: int = 5,
) -> list[dict[str, Any]]:
    return [
        _node(
            generate_id + 1,
            "PreviewAudio",
            [1220, 100],
            [300, 100],
            order,
            inputs=[_input("audio", "AUDIO", preview_link)],
            outputs=[_output("audio", "AUDIO", None, 0)],
            title=f"{step} 试听",
            core=True,
            color="#33505d",
            bgcolor="#1d3039",
        ),
        _node(
            generate_id + 2,
            "SaveAudio",
            [1220, 270],
            [330, 120],
            order + 1,
            inputs=[_input("audio", "AUDIO", save_link)],
            outputs=[_output("audio", "AUDIO", None, 0)],
            widgets=[prefix],
            title=f"{step + 1} 保存音频",
            core=True,
            color="#33505d",
            bgcolor="#1d3039",
        ),
    ]


def _workflow(filename: str, nodes: list[dict[str, Any]], links: list[list[Any]], title: str) -> dict[str, Any]:
    node_ids = [int(node["id"]) for node in nodes]
    link_ids = [int(link[0]) for link in links]
    return {
        "id": str(UUID(WORKFLOW_IDS[filename])),
        "revision": 0,
        "last_node_id": max(node_ids),
        "last_link_id": max(link_ids),
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": f"{title}｜先在①中阅读并勾选 accept_model_license",
                "bounding": [-30, 20, 1610, 750],
                "color": "#ff6f9f",
                "font_size": 22,
                "flags": {},
            },
            {
                "id": 2,
                "title": INLINE_EVENT_GUIDE,
                "bounding": [390, 590, 790, 115],
                "color": "#ff8fbd",
                "font_size": 18,
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.82, "offset": [100, 80]},
            "frontendVersion": "1.24.4",
            "t8_example_kind": "ui_workflow",
            "t8_node_version": PACKAGE_VERSION,
        },
        "version": 0.4,
    }


def voice_design() -> dict[str, Any]:
    nodes = [
        _loader(1),
        _node(
            2,
            "T8_BreezeTTS_DesignRequest",
            [390, 80],
            [390, 250],
            1,
            outputs=[_output("request", "BREEZE_T8_REQUEST", [2], 0)],
            widgets=[
                "[笑] 欢迎使用 Breeze TTS 2。",
                "一位温柔自信的年轻女性，声音清晰，语气亲切。",
                4.0,
            ],
            title="② 声音设计 · text 支持 [笑]/(laugh)",
            color="#6f3d65",
            bgcolor="#342033",
        ),
        _settings(3, 3),
        _generate(4, 1, 2, 3, [4, 5], 3),
        *_audio_outputs(4, 4, 5, "audio/T8_Breeze_design", 4),
    ]
    links = [
        [1, 1, 0, 4, 0, "BREEZE_T8_MODEL"],
        [2, 2, 0, 4, 1, "BREEZE_T8_REQUEST"],
        [3, 3, 0, 4, 2, "BREEZE_T8_SETTINGS"],
        [4, 4, 0, 6, 0, "AUDIO"],
        [5, 4, 0, 5, 0, "AUDIO"],
    ]
    return _workflow("voice_design_workflow.json", nodes, links, "Breeze TTS 2 声音设计")


def _reference_workflow(*, direction: bool) -> dict[str, Any]:
    request_type = "T8_BreezeTTS_DirectionRequest" if direction else "T8_BreezeTTS_CloneRequest"
    filename = "voice_direction_workflow.json" if direction else "voice_clone_workflow.json"
    label = "声音导演" if direction else "声音克隆"
    widgets: list[Any]
    if direction:
        widgets = [
            "[清嗓子] 我们需要认真讨论一下昨晚发生的事情。",
            "请替换为参考音频的准确逐字稿。",
            "语速放慢，语气克制而严肃。",
            4.0,
        ]
    else:
        widgets = [
            "[叹气] 很高兴再次听到你的声音。",
            "请替换为参考音频的准确逐字稿。",
            "Speak clearly and naturally.",
            1.0,
        ]
    nodes = [
        _loader(1),
        _node(
            2,
            "LoadAudio",
            [390, 50],
            [360, 160],
            1,
            outputs=[_output("AUDIO", "AUDIO", [2], 0)],
            widgets=["reference.wav", None, ""],
            title="② 选择参考音频",
            core=True,
            color="#33505d",
            bgcolor="#1d3039",
        ),
        _node(
            3,
            request_type,
            [390, 250],
            [420, 310],
            2,
            inputs=[_input("reference_audio", "AUDIO", 2)],
            outputs=[_output("request", "BREEZE_T8_REQUEST", [3], 0)],
            widgets=widgets,
            title=f"③ {label} · text 支持行内声音事件",
            color="#6f3d65",
            bgcolor="#342033",
        ),
        _settings(4, 4, order=3, step=4),
        _generate(5, 1, 3, 4, [5, 6], 4, step=5),
        *_audio_outputs(
            5,
            5,
            6,
            f"audio/T8_Breeze_{'direction' if direction else 'clone'}",
            5,
            step=6,
        ),
    ]
    links = [
        [1, 1, 0, 5, 0, "BREEZE_T8_MODEL"],
        [2, 2, 0, 3, 0, "AUDIO"],
        [3, 3, 0, 5, 1, "BREEZE_T8_REQUEST"],
        [4, 4, 0, 5, 2, "BREEZE_T8_SETTINGS"],
        [5, 5, 0, 7, 0, "AUDIO"],
        [6, 5, 0, 6, 0, "AUDIO"],
    ]
    return _workflow(filename, nodes, links, f"Breeze TTS 2 {label}")


def voice_bundle() -> dict[str, Any]:
    nodes = [
        _loader(1),
        _node(
            2,
            "T8_BreezeTTS_VoiceBundleRequest",
            [390, 80],
            [420, 330],
            1,
            outputs=[
                _output("request", "BREEZE_T8_REQUEST", [2], 0),
                _output("reference_audio", "AUDIO", None, 1),
                _output("voice_info", "STRING", None, 2),
            ],
            widgets=[
                "D:/voices/my-role.t8voice.zip",
                "[咳嗽] 这句台词使用桌面版导出的音色。",
                "override",
                "情绪温暖，语速自然，在句末轻微停顿。",
                0.0,
            ],
            title="② 桌面音色包 · text 支持行内声音事件",
            color="#6f3d65",
            bgcolor="#342033",
        ),
        _settings(3, 3),
        _generate(4, 1, 2, 3, [4, 5], 3),
        *_audio_outputs(4, 4, 5, "audio/T8_Breeze_voice_bundle", 4),
    ]
    links = [
        [1, 1, 0, 4, 0, "BREEZE_T8_MODEL"],
        [2, 2, 0, 4, 1, "BREEZE_T8_REQUEST"],
        [3, 3, 0, 4, 2, "BREEZE_T8_SETTINGS"],
        [4, 4, 0, 6, 0, "AUDIO"],
        [5, 4, 0, 5, 0, "AUDIO"],
    ]
    return _workflow("voice_bundle_workflow.json", nodes, links, "Breeze TTS 2 桌面音色包")


def main() -> None:
    workflows = {
        "voice_design_workflow.json": voice_design(),
        "voice_clone_workflow.json": _reference_workflow(direction=False),
        "voice_direction_workflow.json": _reference_workflow(direction=True),
        "voice_bundle_workflow.json": voice_bundle(),
    }
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, workflow in workflows.items():
        (EXAMPLES_DIR / filename).write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
