from __future__ import annotations

import json
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class UIWorkflowTests(unittest.TestCase):
    def test_frontend_workflows_have_valid_nodes_and_links(self) -> None:
        expected = {
            "voice_design_workflow.json",
            "voice_clone_workflow.json",
            "voice_direction_workflow.json",
            "voice_bundle_workflow.json",
        }
        paths = list((PACKAGE_DIR / "examples").glob("*_workflow.json"))
        self.assertEqual({path.name for path in paths}, expected)

        allowed_types = {
            "T8_BreezeTTS_ModelLoader",
            "T8_BreezeTTS_DesignRequest",
            "T8_BreezeTTS_CloneRequest",
            "T8_BreezeTTS_DirectionRequest",
            "T8_BreezeTTS_VoiceBundleRequest",
            "T8_BreezeTTS_LineDirection",
            "T8_BreezeTTS_GenerationSettings",
            "T8_BreezeTTS_Generate",
            "LoadAudio",
            "PreviewAudio",
            "SaveAudio",
        }
        required_types = {
            "T8_BreezeTTS_ModelLoader",
            "T8_BreezeTTS_Generate",
            "PreviewAudio",
            "SaveAudio",
        }

        for path in paths:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(workflow["version"], 0.4)
            self.assertEqual(workflow["extra"]["t8_example_kind"], "ui_workflow")
            nodes = {node["id"]: node for node in workflow["nodes"]}
            self.assertEqual(len(nodes), len(workflow["nodes"]))
            self.assertTrue(required_types <= {node["type"] for node in nodes.values()})
            self.assertTrue({node["type"] for node in nodes.values()} <= allowed_types)

            for core_type in ("PreviewAudio", "SaveAudio"):
                core_node = next(node for node in nodes.values() if node["type"] == core_type)
                self.assertEqual(core_node["outputs"][0]["type"], "AUDIO")

            for link_id, source_id, source_slot, target_id, target_slot, data_type in workflow["links"]:
                source = nodes[source_id]
                target = nodes[target_id]
                self.assertEqual(source["outputs"][source_slot]["type"], data_type)
                self.assertIn(link_id, source["outputs"][source_slot]["links"])
                self.assertEqual(target["inputs"][target_slot]["type"], data_type)
                self.assertEqual(target["inputs"][target_slot]["link"], link_id)

            api_path = PACKAGE_DIR / "examples" / path.name.replace("_workflow.json", "_api.json")
            api_prompt = json.loads(api_path.read_text(encoding="utf-8"))
            self.assertNotIn("nodes", api_prompt)
            self.assertNotIn("links", api_prompt)


if __name__ == "__main__":
    unittest.main()
