import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json


def inspect(settings_path, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = read_json(settings_path)
    base = Path(settings['development_base'])
    config = read_json(base / 'config.json')
    rows = []
    for video in settings['development_videos']:
        with np.load(base / video / 'indices.npz', allow_pickle=False) as saved:
            offsets = saved['offsets'].copy()
            frame_indices = saved['frame_indices'].copy()
        available = np.load(Path(config['descriptors']) / video / 'available.npy')
        for episode in read_json(base / video / 'summary.json')['episodes']:
            if not episode['click']['available']:
                continue
            folder = Path(config['episodes']) / video / 'sources' / episode['source_lesion_id']
            click = read_json(folder / 'click_input.json')
            with np.load(folder / 'tracks.npz', allow_pickle=False) as saved:
                assert np.array_equal(offsets, saved['offsets'])
                assert np.array_equal(frame_indices, saved['frame_indices'])
                tracks = saved['track_ids'].copy()
            with np.load(folder / 'decisions.npz', allow_pickle=False) as saved:
                data = {key: saved[key].copy() for key in saved.files}
            position = offsets[click['output_index']] + click['detection_index']
            identity = int(tracks[position])
            assert identity == click['track_binding']['track_id'] and identity >= 0
            observed = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
            candidates = np.flatnonzero((tracks == identity) & available & (observed > click['output_index']))
            column = list(data['lesion_ids']).index(episode['source_lesion_id'])
            eligible = data['post_click'] & data['known'] & data['baseline'][:, column]
            duration = data['end'] - data['start']
            strata = {name: float(duration[eligible & (data['source_track_group'] == name)].sum())
                      for name in ('clicked_track', 'other_track', 'unassigned')}
            row = dict(video=video, episode_id=episode['episode_id'], source_available=bool(available[position]),
                       later_clicked_track_descriptors=len(candidates),
                       first_update_frame=int(frame_indices[observed[candidates[0]]]) if len(candidates) else None,
                       last_update_frame=int(frame_indices[observed[candidates[-1]]]) if len(candidates) else None,
                       source_prompt_seconds=strata)
            rows.append(row)
            print('MEMORY_OPPORTUNITY', row['episode_id'], len(candidates), strata, flush=True)
    result = dict(status='COMPLETE', population='previously_examined_development', episodes=rows,
                  sources_with_later_available_track_observations=sum(row['later_clicked_track_descriptors'] > 0 for row in rows),
                  sources=len(rows), settings=str(settings_path),
                  interpretation='Descriptive opportunity counts. Tracker assignment does not establish true identity or memory-update benefit.')
    atomic_write_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.settings, args.output)
