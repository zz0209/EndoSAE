import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from pathlib import Path

import numpy as np

from encode_token_causal_identity import video_inputs
from src.checkpoint_io import atomic_write_json, read_json
from src.evaluation.realcolon_task import digest


def select(run):
    config = read_json(run / 'config.json')
    reference = read_json(Path(config['prefix_run']) / 'config.json')
    application = read_json(Path(reference['application_reference']) / 'config.json')
    root = Path(reference['event_reference']) / 'inputs'
    result = {}
    for item in read_json(root / 'summary.json')['videos']:
        video = item['video']
        manifest = read_json(root / video / 'events.json')
        _, _, records, tracks, available, _, _, files = video_inputs(application, video)
        with np.load(root / video / 'tokens.npz', allow_pickle=False) as saved:
            current = saved['positions'].copy()
        offsets = tracks['offsets']
        lag = [round(manifest['fps'] * seconds) for seconds in config['history_seconds']]
        assert lag[0] >= 8 and lag[1] - lag[0] >= 8
        rows, required = [], set()
        for position in current:
            output = int(np.searchsorted(offsets[1:], position, side='right'))
            identity = tracks['track_ids'][position]
            previous = []
            for shift in lag:
                past = output - shift
                if past < 0 or identity < 0:
                    break
                a, b = offsets[past:past + 2]
                matches = np.flatnonzero(tracks['track_ids'][a:b] == identity) + a
                assert len(matches) <= 1
                if not len(matches) or not available[matches[0]]:
                    break
                previous.append(int(matches[0]))
                assert records[past]['frame_index'] < records[output]['frame_index']
            complete = len(previous) == len(lag)
            if complete:
                required.update(previous)
            rows.append(dict(position=int(position), output=output, track_id=int(identity),
                complete=complete, previous=previous if complete else []))
        extra = sorted(required - set(current.tolist()))
        assert extra and np.all(available[extra])
        source = int(manifest['episodes'][0]['source_position'])
        result[video] = dict(rows=rows, positions=extra, smoke_positions=[source, extra[0]],
            original_positions=len(current), complete_histories=sum(r['complete'] for r in rows),
            additional_positions=len(extra), lag_frames=lag, fps=manifest['fps'],
            inputs={str(p): digest(p) for p in files},
            event_manifest_sha256=digest(root / video / 'events.json'))
        print('QUERY_HISTORY_SELECTION', video, len(current), len(extra), result[video]['complete_histories'], flush=True)
    atomic_write_json(run / 'history_selection.json', dict(status='COMPLETE', videos=result,
        source_sha256=digest(__file__), config_sha256=digest(run / 'config.json'),
        rule='Same causal detector track at exactly 0.5 and 1.0 seconds earlier, rounded to frames. Both observations must be available. No image labels used for history selection.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    select(parser.parse_args().run)
