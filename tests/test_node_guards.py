from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "comfyui_breeze_tts_t8_guard_tests",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
package = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)
loader = sys.modules[f"{SPEC.name}.loader"]
nodes = sys.modules[f"{SPEC.name}.nodes"]
model_integrity = sys.modules[f"{SPEC.name}.model_integrity"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors(path: Path, tensor_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps(
        {tensor_name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0")


def _finish_snapshot(root: Path) -> None:
    for relative in model_integrity.REQUIRED_JSON_FILES:
        _write_json(root / relative, {"valid": True})
    _write_safetensors(root / model_integrity.REQUIRED_CODEC_WEIGHTS, "codec.weight")
    _write_safetensors(root / "model-00001-of-00001.safetensors", "model.weight")
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}},
    )


class NodeGuardTests(unittest.TestCase):
    def test_partial_snapshot_continues_download_and_is_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            partial = base / loader._safe_repo_name(loader.MODEL_REPO_ID)
            _write_json(partial / "config.json", {"partial": True})

            def finish_download(**kwargs):
                self.assertEqual(kwargs["revision"], loader.MODEL_REVISION)
                self.assertEqual(Path(kwargs["local_dir"]), partial)
                _finish_snapshot(partial)
                return str(partial)

            with (
                mock.patch.object(loader, "model_dirs", return_value=[base]),
                mock.patch("huggingface_hub.snapshot_download", side_effect=finish_download) as download,
            ):
                resolved, weights_name = loader.resolve_model_dir(loader.BF16_LABEL, True)
            self.assertEqual(resolved, partial)
            self.assertEqual(weights_name, "model.safetensors.index.json")
            download.assert_called_once()
            self.assertTrue(loader._model_file_report(partial, weights_name).complete)

    def test_reference_duration_rejected_before_codec_encode(self):
        bundle = SimpleNamespace()
        reference = {
            "waveform": torch.zeros(1, 1, 601),
            "sample_rate": 10,
        }
        request = {
            "text": "test",
            "reference_audio": reference,
            "reference_text": "reference",
        }
        with (
            mock.patch.object(loader, "resume_bundle_to_device", return_value=None),
            mock.patch.object(
                nodes.runtime,
                "encode_reference_audio",
                side_effect=AssertionError("codec encode must not run"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "超过 60 秒上限"):
                nodes._generate_audio(bundle, request, {})

    def test_unload_waits_for_generation_lock(self):
        self.assertIs(nodes._GENERATION_LOCK, loader.GENERATION_LOCK)
        called = threading.Event()

        def fake_unload(*_args, **_kwargs):
            called.set()

        nodes._GENERATION_LOCK.acquire()
        try:
            with mock.patch.object(loader, "_unload_breeze_bundle_locked", side_effect=fake_unload):
                worker = threading.Thread(target=loader.unload_breeze_bundle, args=(object(),))
                worker.start()
                time.sleep(0.05)
                self.assertFalse(called.is_set(), "unload crossed the active generation lock")
                nodes._GENERATION_LOCK.release()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(called.is_set())
        finally:
            if nodes._GENERATION_LOCK.locked():
                nodes._GENERATION_LOCK.release()

    def test_same_generation_thread_cannot_unload_bundle(self):
        self.assertTrue(loader.try_begin_generation())
        try:
            with self.assertRaisesRegex(RuntimeError, "正在生成的同一线程"):
                loader.unload_breeze_bundle(object())
        finally:
            loader.end_generation()

    def test_clone_unload_hook_ignores_unrelated_models(self):
        model = object()
        codec = object()
        patcher = SimpleNamespace(model=model)
        bundle = SimpleNamespace(model=model, codec=codec, patchers=[patcher])
        unrelated = SimpleNamespace(model=object())
        self.assertFalse(loader._unload_request_targets_bundle(bundle, (unrelated,), {}))
        self.assertTrue(loader._unload_request_targets_bundle(bundle, (patcher,), {}))
        self.assertTrue(loader._unload_request_targets_bundle(bundle, (SimpleNamespace(model=patcher),), {}))


if __name__ == "__main__":
    unittest.main()
