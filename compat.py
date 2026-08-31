"""Dependency compatibility checks that run before the heavyweight model imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from importlib.util import find_spec

from packaging.version import InvalidVersion, Version


MIN_TRANSFORMERS = Version("4.57.0")
MAX_TRANSFORMERS = Version("6.0.0")
TESTED_TRANSFORMERS = ("4.57.3", "5.16.1")


@dataclass(frozen=True)
class CompatibilityReport:
    transformers: str
    supported: bool
    message: str
    tested: tuple[str, ...] = TESTED_TRANSFORMERS

    def to_dict(self) -> dict:
        return asdict(self)


def check_transformers(*, raise_on_error: bool = True) -> CompatibilityReport:
    if find_spec("numpy") is None:
        message = (
            "ComfyUI 宿主缺少 NumPy；请修复宿主环境。T8 节点不会自行安装或覆盖 NumPy。"
        )
        report = CompatibilityReport("unknown", False, message)
        if raise_on_error:
            raise RuntimeError(message)
        return report
    try:
        raw = metadata.version("transformers")
        version = Version(raw)
    except metadata.PackageNotFoundError as exc:
        report = CompatibilityReport("missing", False, "未安装 transformers；请使用 ComfyUI 自带环境安装。")
        if raise_on_error:
            raise RuntimeError(report.message) from exc
        return report
    except InvalidVersion as exc:
        report = CompatibilityReport(str(exc), False, "无法识别 transformers 版本。")
        if raise_on_error:
            raise RuntimeError(report.message) from exc
        return report

    supported = MIN_TRANSFORMERS <= version < MAX_TRANSFORMERS
    message = (
        f"Transformers {raw} 兼容；T8 节点内置 4.57 与 5.x API 适配。"
        if supported
        else f"Transformers {raw} 不受支持；需要 >=4.57,<6。请勿单独覆盖 ComfyUI 的 Torch。"
    )
    report = CompatibilityReport(raw, supported, message)
    if raise_on_error and not supported:
        raise RuntimeError(message)
    return report
