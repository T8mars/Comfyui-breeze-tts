"""comfyui-breeze-tts-T8 entrypoint."""

from __future__ import annotations

import logging

from .compat import check_transformers

__version__ = "0.1.0"
logger = logging.getLogger("BreezeTTS2T8")

report = check_transformers(raise_on_error=True)
logger.info("%s", report.message)

from .loader import register_model_folder  # noqa: E402
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

register_model_folder()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

