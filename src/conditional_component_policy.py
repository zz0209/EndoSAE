import hashlib
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.checkpoint_io import read_json


SUPPORT_NAMES = ["observed_frames", "log_token_count", "frame_weight_concentration",
                 "frame_count_cv", "row_spread", "column_spread", "centroid_motion"]
FEATURE_NAMES = SUPPORT_NAMES + ["cohort_mean", "cohort_std", "cohort_q90", "cohort_q99",
    "cohort_max", "self_cosine", "relative_edit_norm", "log_raw_norm",
    "code_concentration", "active_fraction", "action_fraction", "action_strength"]
REFERENCE_FEATURES = [i for i, name in enumerate(FEATURE_NAMES)
                      if name not in ("code_concentration", "active_fraction", "action_fraction", "action_strength")]


def mask_features(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (8, 196) or not mask.any():
        raise ValueError("A source must have observed eight-frame-grid support")
    count = mask.sum(1).astype(float)
    observed = count > 0
    weights = count / count.sum()
    frame, position = np.nonzero(mask)
    row, column = position // 14, position % 14
    centroids = np.array([[row[frame == f].mean(), column[frame == f].mean()]
                          for f in np.flatnonzero(observed)]) / 13
    motion = np.linalg.norm(np.diff(centroids, axis=0), axis=1).mean() if len(centroids) > 1 else 0.
    return np.array([observed.mean(), np.log1p(count.sum()), weights @ weights,
                     count[observed].std() / count[observed].mean(), row.std() / 13,
                     column.std() / 13, motion], dtype=np.float64)


def cohort_weights(records, indices):
    videos = sorted({records[i]["video_id"] for i in indices})
    weights = np.empty(len(indices), dtype=np.float64)
    for video in videos:
        lesions = sorted({records[i]["lesion_id"] for i in indices if records[i]["video_id"] == video})
        for lesion in lesions:
            locations = [k for k, i in enumerate(indices) if records[i]["lesion_id"] == lesion]
            weights[locations] = 1 / (len(videos) * len(lesions) * len(locations))
    if not len(weights) or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid procedure/lesion-weighted cohort")
    return weights


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    return values[order[np.minimum(np.searchsorted(cdf, quantiles), len(order) - 1)]]


def candidate_features(original, edited, codes, memories, cohort, weights, support, candidates):
    original, edited = np.asarray(original), np.asarray(edited)
    count = len(candidates)
    if memories.shape != (count, 128) or edited.shape != (count, 768):
        raise ValueError("Candidate arrays differ from the fixed action budget")
    similarities = memories.astype(np.float64) @ cohort.astype(np.float64).T
    mean = similarities @ weights
    std = np.sqrt(((similarities - mean[:, None]) ** 2) @ weights)
    quantiles = np.array([weighted_quantiles(row, weights, [.9, .99]) for row in similarities])
    norm = np.linalg.norm(original)
    if norm <= 0 or not np.isfinite(similarities).all():
        raise ValueError("Invalid source or cohort scores")
    amplitude = np.abs(codes).astype(np.float64)
    concentration = float((amplitude @ amplitude) / max(amplitude.sum() ** 2, np.finfo(float).tiny))
    dynamic = np.column_stack([mean, std, quantiles, similarities.max(1),
        memories.astype(float) @ memories[0].astype(float), np.linalg.norm(edited - original, axis=1) / norm,
        np.full(count, np.log(norm)), np.full(count, concentration), np.full(count, np.mean(amplitude > 0)),
        np.array([c["k"] for c in candidates]) / len(codes), np.array([c["strength"] for c in candidates])])
    result = np.column_stack([np.broadcast_to(support, (count, len(support))), dynamic])
    if result.shape != (count, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid available-context features")
    return result


def contrast_features(features):
    features = np.asarray(features)
    difference = features - features[..., :1, :]
    norm_index = FEATURE_NAMES.index("relative_edit_norm")
    interactions = features[..., :len(SUPPORT_NAMES)] * difference[..., norm_index, None]
    return np.concatenate([difference, interactions], axis=-1)


def source_weights(metadata):
    weights = np.empty(len(metadata), dtype=np.float64)
    videos = sorted({r["video_id"] for r in metadata})
    for video in videos:
        sources = sorted({r["source_index"] for r in metadata if r["video_id"] == video})
        for source in sources:
            positions = [i for i, r in enumerate(metadata) if r["video_id"] == video and r["source_index"] == source]
            weights[positions] = 1 / (len(videos) * len(sources) * len(positions))
    return weights


def fit_linear(x, y, weights, alpha, contrast=False):
    scaler = StandardScaler(with_mean=not contrast)
    scaled = scaler.fit_transform(x, sample_weight=weights)
    model = Ridge(alpha=alpha, fit_intercept=not contrast, solver="svd")
    model.fit(scaled, y, sample_weight=weights * len(weights))
    coefficient = model.coef_ / scaler.scale_
    intercept = float(model.intercept_) - float(coefficient @ scaler.mean_) if not contrast else 0.
    exported = dict(coefficient=coefficient.tolist(), intercept=intercept,
                    alpha=alpha, contrast=contrast, features=x.shape[1], samples=len(x))
    if not np.allclose(predict_linear(exported, x), model.predict(scaled), atol=1e-10, rtol=1e-10):
        raise ValueError("Ridge export differs from sklearn prediction")
    return exported


def predict_linear(model, features):
    return np.asarray(features) @ np.asarray(model["coefficient"]) + model["intercept"]


def fit_policy(features, thresholds, utility, metadata, alpha):
    weights = source_weights(metadata)
    count = features.shape[1]
    advantage = utility - utility[:, :1]
    contrasts = contrast_features(features)
    reference_threshold = fit_linear(features[:, 0, REFERENCE_FEATURES], thresholds[:, 0], weights, alpha)
    threshold = fit_linear(contrasts[:, 1:].reshape(-1, contrasts.shape[-1]),
        (thresholds - thresholds[:, :1])[:, 1:].reshape(-1),
        np.repeat(weights / (count - 1), count - 1), alpha, contrast=True)
    effect = fit_linear(contrasts[:, 1:].reshape(-1, contrasts.shape[-1]), advantage[:, 1:].reshape(-1),
        np.repeat(weights / (count - 1), count - 1), alpha, contrast=True)
    mean_advantage = weights @ advantage
    fixed = int(np.argmax(mean_advantage))
    return dict(reference_threshold=reference_threshold, threshold=threshold, advantage=effect, fixed_action=fixed,
                fitting_mean_advantage=mean_advantage.tolist(), feature_names=FEATURE_NAMES,
                fitting_procedures=sorted({r["video_id"] for r in metadata}))


def choose(policy, features):
    advantage = predict_linear(policy["advantage"], contrast_features(features))
    if not np.array_equal(advantage[..., 0], np.zeros_like(advantage[..., 0])):
        raise ValueError("Zero-action advantage must be exactly zero")
    action = np.argmax(advantage, axis=-1)
    base = predict_linear(policy["reference_threshold"], features[..., 0, REFERENCE_FEATURES])
    threshold = np.clip(base[..., None] + predict_linear(policy["threshold"], contrast_features(features)), -1., 1.)
    return action, advantage, threshold


def supervised_targets(memories, queries, positive, negative, quantile, temperature):
    if not len(positive) or not len(negative):
        raise ValueError("Tail targets require both chronological identity classes")
    scores = memories.astype(np.float64) @ queries.astype(np.float64).T
    threshold = np.quantile(scores[:, negative], quantile, axis=1)
    utility = expit((scores[:, positive] - threshold[:, None]) / temperature).mean(1)
    return threshold, utility, scores


def raw_key(raw):
    return hashlib.sha256(np.asarray(raw, dtype="<f8").reshape(-1).tobytes()).hexdigest()


class SourcePolicyPredictor:
    def __init__(self, reference, memories, calibrated):
        self.reference = reference
        self.memories = memories
        self.calibrated = calibrated

    def encode(self, raw, batch_size=512):
        values = np.asarray(raw).reshape(-1, 768)
        return np.concatenate([self.reference(values[i:i + batch_size])["supcon_l2"]
                               for i in range(0, len(values), batch_size)])

    def memory(self, raw):
        stored, memory, threshold = self.memories[raw_key(raw)]
        if not np.array_equal(np.asarray(raw).reshape(-1), stored):
            raise ValueError("Source lookup differs from the saved descriptor")
        return np.concatenate([memory, [threshold]]) if self.calibrated else memory.copy()

    def score_encoded(self, memory, codes):
        cosine = np.clip(np.asarray(codes).astype(np.float64) @ memory[:128].astype(np.float64), -1., 1.)
        return 0.5 + 0.25 * (cosine - memory[128]) if self.calibrated else cosine

    def score(self, raw, queries, batch_size=512):
        return self.score_encoded(self.memory(raw), self.encode(queries, batch_size))
