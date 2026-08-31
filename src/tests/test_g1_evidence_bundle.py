import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.g1_evidence_bundle import G1EvidenceBundleError, validate_g1_evidence_bundle


H = "a" * 64


class G1EvidenceBundleTests(unittest.TestCase):
    def _write(self, root, name, payload):
        path = root / name
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"path": name, "sha256": digest}

    def _bundle(self, root):
        pre = self._write(root, "pre.json", {
            "schema_version": "endosae.checkpoint-preflight.v0", "status": "pass", "file_sha256": H
        })
        inv = self._write(root, "inv.json", {
            "schema_version": "endosae.checkpoint-inventory.v0", "status": "pass",
            "checkpoint_sha256": H, "preflight_report_sha256": pre["sha256"],
            "behavior_testing_allowed": True
        })
        prep = self._write(root, "prep.json", {"schema_version": "endosae.reference-input-contract.v0"})
        sampling = self._write(root, "sampling.json", {"schema_version": "endosae.sampling-index.v0"})
        prep_runtime = self._write(root, "prep-runtime.json", {
            "schema_version": "endosae.preprocessing-runtime-fixture.v0", "status": "pass",
            "reference_contract_sha256": prep["sha256"],
            "sampling_index_asset_sha256": sampling["sha256"]
        })
        temporal = self._write(root, "temporal.json", {
            "schema_version": "endosae.temporal-coordinate-fixture.v0", "status": "pass",
            "checkpoint_sha256": H, "sampling_index_asset_sha256": sampling["sha256"]
        })
        hooks = [self._write(root, f"hook-{i}.json", {
            "schema_version": "endosae.hook-fixture.v1", "status": "pass",
            "checkpoint_sha256": H, "fixture_id": f"layer-{i}"
        }) for i in range(3)]
        return {
            "schema_version": "endosae.g1-evidence-bundle.v0", "status": "pass",
            "checkpoint_sha256": H, "preflight_report": pre, "inventory_report": inv,
            "preprocessing_contract": prep, "preprocessing_runtime_fixture": prep_runtime,
            "hook_fixtures": hooks,
            "temporal_coordinate_fixture": temporal, "sampling_index_asset": sampling,
            "behavior_testing_allowed": True, "formal_activation_cache_allowed": True,
        }

    def test_valid_hash_chain_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_g1_evidence_bundle(self._bundle(root), root)

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            (root / "sampling.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(G1EvidenceBundleError, "hash mismatch"):
                validate_g1_evidence_bundle(bundle, root)

    def test_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            bundle["checkpoint_sha256"] = "b" * 64
            with self.assertRaisesRegex(G1EvidenceBundleError, "preflight checkpoint"):
                validate_g1_evidence_bundle(bundle, root)

    def test_source_contract_cannot_replace_runtime_preprocessing_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            runtime_path = root / bundle["preprocessing_runtime_fixture"]["path"]
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            payload["status"] = "invalid"
            bundle["preprocessing_runtime_fixture"] = self._write(root, "prep-runtime-invalid.json", payload)
            with self.assertRaisesRegex(G1EvidenceBundleError, "runtime fixture must pass"):
                validate_g1_evidence_bundle(bundle, root)

    def test_cache_cannot_be_enabled_on_failed_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            bundle["status"] = "fail"
            with self.assertRaisesRegex(G1EvidenceBundleError, "formal cache"):
                validate_g1_evidence_bundle(bundle, root)


if __name__ == "__main__":
    unittest.main()
