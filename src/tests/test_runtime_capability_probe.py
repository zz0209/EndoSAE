import unittest

from src.runtime_capability_probe import PACKAGES, collect_runtime_capability


class RuntimeCapabilityProbeTests(unittest.TestCase):
    def test_probe_is_explicitly_non_checkpoint_evidence(self):
        report = collect_runtime_capability("unit-test-current-python")
        self.assertEqual(report["schema_version"], "endosae.runtime-capability-probe.v0")
        self.assertEqual(report["status"], "probe-only")
        self.assertFalse(report["checkpoint_accessed"])
        self.assertEqual(set(report["packages"]), set(PACKAGES))


if __name__ == "__main__":
    unittest.main()
