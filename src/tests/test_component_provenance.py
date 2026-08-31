import json
import unittest
from pathlib import Path

from src.component_provenance import ProvenanceError, validate_component_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "endofm_component_provenance_v0.json"


class ComponentProvenanceTests(unittest.TestCase):
    def registry(self):
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_pinned_registry_is_conservative(self):
        validate_component_registry(self.registry())

    def test_restrictive_component_cannot_be_marked_clear(self):
        registry = self.registry()
        timesformer = next(c for c in registry["components"] if c["component_id"] == "timesformer")
        timesformer["redistribution_status"] = "cleared"
        with self.assertRaisesRegex(ProvenanceError, "timesformer"):
            validate_component_registry(registry)

    def test_required_component_cannot_silently_disappear(self):
        registry = self.registry()
        registry["components"] = [c for c in registry["components"] if c["component_id"] != "stft"]
        with self.assertRaisesRegex(ProvenanceError, "stft"):
            validate_component_registry(registry)


if __name__ == "__main__":
    unittest.main()
