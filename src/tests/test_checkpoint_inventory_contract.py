import json
import unittest

from checkpoint_inventory_contract import (
    PROJECT_ROOT,
    CheckpointInventoryContractError,
    validate_inventory_contract,
)


class CheckpointInventoryContractTests(unittest.TestCase):
    def setUp(self):
        path = PROJECT_ROOT / "configs/isolated_checkpoint_inventory_contract_v0.json"
        self.record = json.loads(path.read_text(encoding="utf-8"))

    def test_frozen_contract(self):
        validate_inventory_contract(self.record)

    def test_rejects_network(self):
        changed = json.loads(json.dumps(self.record))
        changed["required_loader"]["network_enabled"] = True
        with self.assertRaises(CheckpointInventoryContractError):
            validate_inventory_contract(changed)

    def test_rejects_missing_hard_stop(self):
        changed = json.loads(json.dumps(self.record))
        changed["hard_stops"]["weights_only_failure"] = False
        with self.assertRaises(CheckpointInventoryContractError):
            validate_inventory_contract(changed)


if __name__ == "__main__":
    unittest.main()

