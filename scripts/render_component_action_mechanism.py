import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from analyze_component_action_mechanism import settings, transfer, native, read_json, atomic_write_json, digest, pause_after_checkpoint


def image_panel(ax, path, box, title, crop=False):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        width, height = rgb.size
        if crop:
            rgb = rgb.resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
        ax.imshow(rgb)
    if box is not None:
        x1, y1, x2, y2 = box
        if crop:
            x1, x2 = x1 * 280 / width - 28, x2 * 280 / width - 28
            y1, y2 = y1 * 224 / height, y2 * 224 / height
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='#ffdb00', linewidth=1.3))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


@torch.no_grad()
def render(run, smoke):
    config, capacity, event, original, root = settings(run, smoke)
    output = root / 'analysis/images'
    output.mkdir(parents=True, exist_ok=True)
    cases = read_json(root / 'analysis/image_cases.json')
    import pandas as pd
    rows = pd.read_csv(root / 'analysis/events.csv')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = transfer.load_models(dict(original, methods=['token_sparse']), torch.device(config['device']))
    summaries, started = [], time.perf_counter()
    for video in sorted({row['video'] for row in cases}):
        manifest = read_json(event / 'inputs' / video / 'events.json')
        _, _, records, tracks, _, _, _, _ = transfer.video_inputs(original, video)
        with np.load(event / 'inputs' / video / 'tokens.npz') as saved:
            positions, offsets, tokens = [saved[k] for k in ['positions', 'offsets', 'tokens']]
        lookup = {int(p): i for i, p in enumerate(positions)}
        for case in [r for r in cases if r['video'] == video]:
            name = video + '_' + case['role']
            identity = dict(source=digest(__file__), cases=digest(root / 'analysis/image_cases.json'),
                config=digest(run / 'config.json'), event_inputs=digest(event / 'inputs' / video / 'complete.json'))
            receipt_path = output / (name + '.json')
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                assert receipt['identity'] == identity and digest(output / (name + '.png')) == receipt['image_sha256']
                summaries.append(receipt)
                continue
            episode = next(e for e in manifest['episodes'] if e['episode_id'] == case['episode'])
            selected = rows[(rows.model == case['model']) & (rows.video == video) &
                            (rows.episode == case['episode']) & (rows.policy == 'bank_pair') & (rows.action == 'bilateral')]
            query = selected[selected.event_id == case['event_id']].iloc[0]
            companions = selected[selected.after_activation & (selected.same_identity != query.same_identity)].sort_values(['frame', 'event_id'])
            chosen = [('Source', None), (case['role'], query)]
            if len(companions):
                chosen.append(('Same-source identity control', companions.iloc[0]))
            model, mean, scale = models[case['model']]
            with np.load(capacity / 'evaluation' / video / case['model'] / 'effects.npz') as saved:
                reference_lookup = {int(p): i for i, p in enumerate(saved['positions'])}
                reference = saved['unit_codes']
            panels, maximum_error = [], 0.
            for label, row in chosen:
                if row is None:
                    position = episode['source_position']
                    source = Path(original['source_token_root']) / video / 'sources' / case['episode']
                    with np.load(source / 'observed_tokens.npz') as saved:
                        coordinates = saved['positions']
                        raw = saved['tokens']
                    frame_id, box = episode['click']['input_frame'], episode['click']['detection']['xyxy']
                    index = lookup[position]
                    np.testing.assert_array_equal(raw, tokens[offsets[index]:offsets[index + 1]])
                    title = 'Confirmed source'
                else:
                    position = int(row.matched_position)
                    frame = manifest['frames'][str(int(row.output))]
                    detection_index = frame['positions'].index(position)
                    mask, support, reason = native.support(records, tracks, int(row.output), detection_index)
                    assert reason == 'available'
                    coordinates = np.column_stack(np.where(mask))
                    index = lookup[position]
                    raw = tokens[offsets[index]:offsets[index + 1]]
                    frame_id, box = int(row.frame), frame['detections'][detection_index]['xyxy']
                    title = ('Same lesion' if row.same_identity else 'Other lesion') + f' | {label}\n'
                    title += f'Score {row.original_score:.4f} to {row.changed_score:.4f} | retain {row.before} to {row.retained}'
                assert len(coordinates) == len(raw) and coordinates.shape[1] == 2
                normalized = ((raw.astype(float) - mean) / scale).astype(np.float32)
                local = model.encode(torch.from_numpy(normalized).to(config['device'])).cpu().numpy().astype(float)
                unit = local.mean(0)
                unit /= np.linalg.norm(unit)
                error = float(np.max(np.abs(unit - reference[reference_lookup[position]])))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(unit, reference[reference_lookup[position]], atol=2e-6, rtol=1e-5)
                activation = local[:, case['coordinates']].sum(1)
                grid = np.full((8, 196), np.nan)
                grid[coordinates[:, 0], coordinates[:, 1]] = activation
                path = Path(manifest['definition']['frame_root']) / video / f'{frame_id:06d}.jpg'
                panels.append(dict(label=label, path=path, frame=frame_id, box=box, title=title,
                    grid=grid, mass=np.nansum(grid, axis=1), position=position,
                    active_tokens=int(np.count_nonzero(activation)), total_tokens=len(activation)))
            fig, axes = plt.subplots(3, len(panels), figsize=(5 * len(panels), 10), squeeze=False, layout='constrained')
            maximum = max(float(np.nanmax(p['grid'])) for p in panels)
            for column, panel in enumerate(panels):
                image_panel(axes[0, column], panel['path'], panel['box'], panel['title'])
                image_panel(axes[1, column], panel['path'], panel['box'],
                            f"Current-frame activity | {panel['active_tokens']}/{panel['total_tokens']} active support tokens", crop=True)
                values = np.ma.masked_where(~np.isfinite(panel['grid'][7]) | (panel['grid'][7] == 0), panel['grid'][7]).reshape(14, 14)
                heat = axes[1, column].imshow(values, extent=(0, 224, 224, 0), interpolation='nearest',
                    cmap='magma', alpha=.65, vmin=0, vmax=max(maximum, 1e-12))
                axes[2, column].bar(np.arange(-7, 1), panel['mass'], color='#4477aa')
                axes[2, column].set(xlabel='Frame offset from shown frame', ylabel='Sum of coordinate activity',
                                    title='Activity within actual causal support', xticks=np.arange(-7, 1))
            fig.colorbar(heat, ax=list(axes[1]), shrink=.6, label='Selected-coordinate activity; shared scale')
            fig.suptitle(f"{video} | SAE seed {case['seed']} | coordinates {case['coordinates']}\n"
                'Yellow: detector support | Activation location is descriptive', fontsize=12)
            fig.savefig(output / (name + '.png'), dpi=130)
            plt.close(fig)
            receipt = dict(status='COMPLETE', identity=identity, case=case,
                image_sha256=digest(output / (name + '.png')), maximum_unit_error=maximum_error,
                panels=[dict(label=p['label'], frame=p['frame'], position=p['position'], image=str(p['path']),
                    image_sha256=digest(p['path']), detector_box=p['box'], temporal_mass=p['mass'].tolist(),
                    active_tokens=p['active_tokens'], support_tokens=p['total_tokens']) for p in panels],
                display='Original full RGB above; fixed EndoFM resize/center crop below. No color enhancement. Current-frame overlays; full causal-support temporal histogram.',
                interpretation='Activation location and coordinate-intervention effect; input causation and clinical semantics remain unestablished.')
            atomic_write_json(receipt_path, receipt)
            summaries.append(receipt)
            atomic_write_json(output / 'progress.json', dict(completed=len(summaries), total=len(cases), seconds=time.perf_counter() - started))
            print('MECHANISM_IMAGES', len(summaries), '/', len(cases), name, flush=True)
            pause_after_checkpoint(receipt_path)
    atomic_write_json(output / 'summary.json', dict(status='COMPLETE', cases=len(summaries),
        maximum_unit_error=max((r['maximum_unit_error'] for r in summaries), default=0),
        seconds=time.perf_counter() - started, torch=str(torch.__version__), peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    render(args.run, args.smoke)
