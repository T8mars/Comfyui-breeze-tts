from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "breeze_memory_tests", PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)
loader = sys.modules[f"{SPEC.name}.loader"]
native = sys.modules[f"{SPEC.name}.native"]


class MemoryManagementTests(unittest.TestCase):
    def embedding(self):
        embedding = native.T5Gemma2TextScaledWordEmbedding(
            16, 8, 0, embed_scale=2.5, eoi_token_index=15,
        )
        with torch.no_grad():
            embedding.eoi_embedding.fill_(3.0)
        return embedding

    def test_scaled_embedding_preserves_math_and_state_dict(self):
        embedding = self.embedding()
        ids = torch.tensor([[0, 1, 15, 2]])
        expected = embedding(ids).detach()
        keys = tuple(embedding.state_dict())
        native.convert_modules_for_comfy(embedding)
        native.convert_modules_for_comfy(embedding)
        self.assertTrue(embedding.comfy_cast_weights)
        self.assertEqual(tuple(embedding.state_dict()), keys)
        torch.testing.assert_close(embedding(ids), expected)

    def test_scaled_embedding_uses_comfy_cast_and_cleanup_for_paged_weight(self):
        embedding = self.embedding()
        ids = torch.tensor([[1, 15, 2]])
        expected = embedding(ids).detach()
        native.convert_modules_for_comfy(embedding)
        embedding._v = object()
        cast = mock.Mock(return_value=(embedding.weight.clone(), None, "stream"))
        with mock.patch.object(native, "_cast_bias_weight", cast), mock.patch.object(
            native, "_uncast_bias_weight"
        ) as uncast:
            torch.testing.assert_close(embedding(ids), expected)
        cast.assert_called_once()
        self.assertEqual(cast.call_args.kwargs["device"], ids.device)
        self.assertEqual(cast.call_args.kwargs["dtype"], embedding.weight.dtype)
        uncast.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_scaled_embedding_with_cpu_weights_and_cuda_tokens(self):
        embedding = self.embedding()
        ids = torch.tensor([[1, 15, 2]])
        expected = embedding(ids).detach().cuda()
        native.convert_modules_for_comfy(embedding)

        def cast(module, **kwargs):
            return module.weight.to(device=kwargs["device"], dtype=kwargs["dtype"]), None, None

        with mock.patch.object(native, "_cast_bias_weight", cast), mock.patch.object(
            native, "_uncast_bias_weight"
        ):
            result = embedding(ids.cuda())
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(embedding.weight.device.type, "cpu")
        torch.testing.assert_close(result, expected)

    def test_registered_patchers_are_rechecked_after_partial_offload(self):
        patcher = SimpleNamespace(model=object())
        manager = SimpleNamespace(
            current_loaded_models=[SimpleNamespace(model=patcher)], load_models_gpu=mock.Mock()
        )
        with mock.patch.object(loader, "mm", manager):
            loader._register_many_with_comfy([patcher, None])
            loader._register_many_with_comfy([patcher])
        self.assertEqual(manager.load_models_gpu.call_count, 2)
        self.assertEqual(manager.load_models_gpu.call_args.args[0], [patcher])

    def test_unregister_removes_patcher_not_module_from_comfy_registry(self):
        patcher = SimpleNamespace(model=object(), detach=mock.Mock())
        finalizer = mock.Mock()
        loaded = SimpleNamespace(model=patcher, model_finalizer=finalizer)
        unrelated = SimpleNamespace(model=object())
        manager = SimpleNamespace(current_loaded_models=[loaded, unrelated])
        with mock.patch.object(loader, "mm", manager):
            loader._unregister_from_comfy(patcher)
        self.assertEqual(manager.current_loaded_models, [unrelated])
        finalizer.detach.assert_called_once()
        patcher.detach.assert_called_once()

    def test_graph_bundle_requests_full_residency_for_model_and_codec(self):
        patchers = [object(), object()]
        bundle = SimpleNamespace(
            patchers=patchers, decode_mode="cuda_graphs", model=None,
        )
        manager = SimpleNamespace(load_models_gpu=mock.Mock())
        with mock.patch.object(loader, "mm", manager):
            loader.resume_bundle_to_device(bundle)
        manager.load_models_gpu.assert_called_once_with(patchers, force_full_load=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_standalone_cast_handles_offloaded_embedding_without_comfy(self):
        embedding = self.embedding()
        ids = torch.tensor([[1, 15, 2]])
        expected = embedding(ids).detach().cuda()
        native.convert_modules_for_comfy(embedding)
        torch.testing.assert_close(embedding(ids.cuda()), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_codebook_head_casts_cpu_weights_and_cuda_positions(self):
        head = native.BreezeCodebooksHead(8, 4, 16)
        with torch.no_grad():
            head.weight.normal_()
        for positions in (None, torch.tensor([2])):
            with self.subTest(positions=positions):
                states = torch.randn(2, 3 if positions is None else 1, 8)
                indices = range(3) if positions is None else [1]
                expected = torch.stack([
                    torch.nn.functional.linear(states[:, i], head.weight[j].T)
                    for i, j in enumerate(indices)
                ], dim=1).cuda()
                result = head(states.cuda(), positions.cuda() if positions is not None else None)
                torch.testing.assert_close(result, expected)

    def test_graph_weight_reallocation_discards_old_captures(self):
        depth = torch.nn.Linear(8, 8)
        model = SimpleNamespace(depth_decoder=depth, _breeze_depth_runners={"old": object()})
        bundle = SimpleNamespace(
            patchers=[], decode_mode="cuda_graphs", model=model,
            device=torch.device("cpu"), depth_weight_signature=(),
        )
        with mock.patch.object(loader, "_register_many_with_comfy"), mock.patch.object(
            loader, "release_depth_graphs"
        ) as release:
            loader.resume_bundle_to_device(bundle)
            loader.resume_bundle_to_device(bundle)
            release.assert_not_called()
            depth.weight = torch.nn.Parameter(depth.weight.detach().clone())
            loader.resume_bundle_to_device(bundle)
        release.assert_called_once_with(model)
        self.assertEqual(model._breeze_depth_runners, {})
