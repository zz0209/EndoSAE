import numpy as np


METHODS = ('static', 'track_memory', 'reference_memory', 'sparse_guard', 'dense_guard')


def select_writes(offsets, times, available, track_ids, click, reference_scores,
                  guard_scores, thresholds, config, method):
    if method not in METHODS:
        raise ValueError('Unknown memory method')
    if method == 'static':
        return []
    source_position = int(offsets[click['output_index']] + click['detection_index'])
    source_track = int(track_ids[source_position])
    if source_track < 0:
        raise ValueError('Click-initialized track is unavailable')
    candidate = available & (track_ids == source_track)
    if method != 'track_memory':
        candidate = candidate | (available & (reference_scores > thresholds['reference']))
    if method in ('sparse_guard', 'dense_guard'):
        candidate = candidate & (guard_scores[method] > thresholds[method])
    events = []
    last_time = float(click['time'])
    for output in range(click['output_index'] + 1, len(times)):
        if times[output] - last_time < config['update_interval_seconds'] - 1e-9:
            continue
        start, end = offsets[output:output + 2]
        positions = np.flatnonzero(candidate[start:end]) + start
        if not len(positions):
            continue
        selected = int(positions[np.argmax(reference_scores[positions])])
        events.append(dict(output_index=output, detection_position=selected,
                           time=float(times[output]), reference_score=float(reference_scores[selected]),
                           track_id=int(track_ids[selected]), clicked_track=bool(track_ids[selected] == source_track),
                           guard_score=float(guard_scores[method][selected]) if method in guard_scores else None))
        last_time = float(times[output])
    return events


def replay_memory(codes, available, offsets, reference_scores, events, capacity):
    if capacity < 2 or codes.ndim != 2 or len(codes) != len(available):
        raise ValueError('Invalid memory capacity or code shape')
    if len(reference_scores) != len(available) or offsets[-1] != len(available):
        raise ValueError('Memory observations disagree')
    if not np.isfinite(codes[available]).all() or not np.isfinite(reference_scores[available]).all():
        raise ValueError('Nonfinite available identity observation')
    scores = np.asarray(reference_scores, dtype=np.float64).copy()
    winner = np.full(len(scores), -1, dtype=np.int64)
    memory = []
    for number, event in enumerate(events):
        output, position = event['output_index'], event['detection_position']
        if not offsets[output] <= position < offsets[output + 1] or not available[position]:
            raise ValueError('Memory write is outside its observed frame')
        if number and output <= events[number - 1]['output_index']:
            raise ValueError('Memory writes are not strictly chronological')
        memory.append(position)
        memory = memory[-(capacity - 1):]
        begin = offsets[output + 1]
        end = offsets[events[number + 1]['output_index'] + 1] if number + 1 < len(events) else len(scores)
        selected = np.flatnonzero(available[begin:end]) + begin
        if not len(selected):
            continue
        if any(index >= begin for index in memory):
            raise ValueError('Memory contains a current or future observation')
        similarities = np.clip(codes[selected].astype(np.float64) @ codes[memory].astype(np.float64).T, -1., 1.)
        best = similarities.argmax(axis=1)
        best_scores = similarities[np.arange(len(selected)), best]
        improve = best_scores > scores[selected]
        scores[selected[improve]] = best_scores[improve]
        winner[selected[improve]] = np.asarray(memory)[best[improve]]
    if np.any(scores[available] < reference_scores[available]):
        raise ValueError('Immutable initial template was lost')
    return scores, winner


def memory_scores(codes, available, offsets, times, tracks, click, reference,
                  guards, thresholds, config):
    scores, writes, winners = {}, {}, {}
    for method in METHODS:
        events = select_writes(offsets, times, available, tracks, click, reference,
                               guards, thresholds, config, method)
        value, winner = replay_memory(codes, available, offsets, reference, events, config['memory_capacity'])
        scores[method], writes[method], winners[method] = value, events, winner
    return scores, writes, winners
