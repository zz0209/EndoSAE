import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def prepare(run):
    normal, resumed = run / 'smoke_verified', run / 'recovery_verified'
    assert read_json(normal / 'summary.json')['status'] == read_json(resumed / 'summary.json')['status'] == 'COMPLETE'
    arrays = 0
    for path in normal.rglob('*.npz'):
        with np.load(path, allow_pickle=False) as left, np.load(resumed / path.relative_to(normal), allow_pickle=False) as right:
            assert left.files == right.files
            for key in left.files:
                a, b = left[key], right[key]
                assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)
                arrays += 1
    for path in normal.rglob('writes.json'):
        assert read_json(path) == read_json(resumed / path.relative_to(normal))
    assert read_json(normal / 'operating_points.json') == read_json(resumed / 'operating_points.json')
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
                    ('total_physical', 'available_physical', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    atomic_write_json(run / 'verification.json', dict(status='PASS', recovery_arrays_exact=arrays,
        write_events_exact=True, thresholds_exact=True, smoke_seconds=read_json(normal / 'summary.json')['seconds'],
        available_ram_bytes=memory.available_physical, memory_load_percent=memory.load))
    config = read_json(run / 'config.json')
    stages = []
    for seed in config['seeds']:
        folder = run / 'evaluation' / f'seed{seed}'
        stages.append(dict(id=f'evaluate_{seed}', kind='evaluate', label=f'Seed {seed}: complete video replay',
            command=[sys.executable, 'scripts/evaluate_causal_memory_updates.py', '--run', str(run), '--seed', str(seed), '--resume'],
            progress=str(folder / 'progress.json'), output=str(folder / 'summary.json'),
            units=10, unit='procedures', estimate_seconds=240., resources=['disk-e-io', 'disk-d-io']))
    files = ['scripts/evaluate_causal_memory_updates.py', 'src/causal_identity_memory.py', 'scripts/prepare_causal_memory_batch.py',
             'scripts/run_sae_acknowledgement_batch.py']
    atomic_write_json(run / 'batch_plan.json', dict(question=read_json(run / 'protocol.json')['question'], stages=stages,
        source_hashes={path: digest(ROOT / path) for path in files}, config_sha256=digest(run / 'config.json'),
        protocol_sha256=digest(run / 'protocol.json'), estimate_basis='Real two-source smoke; updated from complete procedure counts.'))
    print('RECOVERY_VERIFIED', arrays, 'arrays; available RAM', memory.available_physical, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    prepare(parser.parse_args().run)
