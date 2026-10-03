import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def verify(run):
    output = run / "recovery"
    command = [sys.executable, str(ROOT / "scripts/evaluate_decoder_head_transfer.py"),
               "--run", str(run), "--smoke", "--output", str(output)]
    stopped = subprocess.run(command + ["--stop-after", "1"], capture_output=True, text=True)
    if stopped.returncode != 75:
        raise ValueError(stopped.stdout + stopped.stderr)
    first = next((output / "cells").glob("*/summary.json"))
    signature = file_sha256(first)
    subprocess.run(command + ["--resume"], check=True, stdout=subprocess.DEVNULL)
    if signature != file_sha256(first):
        raise ValueError("Completed cell was modified during recovery")
    count = 0
    for path in (run / "smoke" / "cells").glob("*/*.npz"):
        with np.load(path, allow_pickle=False) as expected, np.load(output / path.relative_to(run / "smoke"), allow_pickle=False) as actual:
            if expected.files != actual.files:
                raise ValueError("Recovered array keys differ")
            for key in expected.files:
                np.testing.assert_array_equal(expected[key], actual[key])
                count += 1
    errors = []
    for path in (run / "smoke" / "cells").glob("*/summary.json"):
        report = read_json(path)
        errors.extend(report["historical_errors"].values())
        for view in report["views"].values():
            for metric, expected in view["means"].items():
                actual = np.mean([p[metric] for p in view["procedures"].values()])
                if abs(actual - expected) > 1e-12:
                    raise ValueError("Equal-procedure aggregation differs")
    if not errors or max(errors) > 1e-12:
        raise ValueError("Historical endpoint verification is absent or failed")
    receipt = dict(status="PASS", arrays_exact=count, first_cell_preserved=True,
                   historical_max_error=max(errors), source_sha256=file_sha256(__file__))
    atomic_write_json(output / "verification.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
