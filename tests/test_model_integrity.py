from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "model_integrity.py"
SPEC = importlib.util.spec_from_file_location("breeze_t8_model_integrity", MODULE_PATH)
model_integrity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = model_integrity
SPEC.loader.exec_module(model_integrity)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_safetensors(path: Path, tensors: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {}
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _create_complete_snapshot(root: Path) -> None:
    for relative in model_integrity.REQUIRED_JSON_FILES:
        _write_json(root / relative, {"valid": True})
    _write_safetensors(root / model_integrity.REQUIRED_CODEC_WEIGHTS, {"codec.weight": 4})
    _write_safetensors(root / "model-00001-of-00002.safetensors", {"model.a": 3})
    _write_safetensors(root / "model-00002-of-00002.safetensors", {"model.b": 5})
    _write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 8},
            "weight_map": {
                "model.a": "model-00001-of-00002.safetensors",
                "model.b": "model-00002-of-00002.safetensors",
            },
        },
    )


class ModelIntegrityTests(unittest.TestCase):
    def test_complete_snapshot_requires_every_indexed_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertTrue(report.complete, report.summary())
            self.assertEqual(
                report.referenced_shards,
                ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
            )

    def test_missing_indexed_shard_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            (root / "model-00002-of-00002.safetensors").unlink()
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertFalse(report.complete)
            self.assertIn("model-00002-of-00002.safetensors", report.missing_files)

    def test_truncated_shard_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            shard = root / "model-00001-of-00002.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertFalse(report.complete)
            self.assertTrue(any("model-00001-of-00002.safetensors" in item for item in report.invalid_files))

    def test_index_tensor_must_exist_in_its_declared_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.a"] = "model-00002-of-00002.safetensors"
            _write_json(index_path, index)
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertFalse(report.complete)
            self.assertTrue(any("索引张量" in item for item in report.invalid_files))

    def test_index_rejects_parent_directory_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.a"] = "../outside.safetensors"
            _write_json(index_path, index)
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertFalse(report.complete)
            self.assertTrue(any("不安全的分片路径" in item for item in report.invalid_files))

    def test_missing_tokenizer_file_is_reported_with_repair_guidance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _create_complete_snapshot(root)
            (root / "tokenizer.json").unlink()
            report = model_integrity.inspect_model_dir(root, "model.safetensors.index.json")
            self.assertIn("tokenizer.json", report.missing_files)
            guidance = model_integrity.repair_guidance(report)
            self.assertIn("download_if_missing=true", guidance)
            self.assertIn(str(root), guidance)


if __name__ == "__main__":
    unittest.main()
