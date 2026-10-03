import numpy as np
from sklearn.svm import LinearSVC
import torch

from src.template_component_adaptation import cohort_tail
from src.token_memory_edit import residual_edit


def frame_views(tokens, mask):
    tokens, mask = np.asarray(tokens), np.asarray(mask, dtype=bool)
    if tokens.shape != (8, 196, 768) or mask.shape != (8, 196) or not mask.any():
        raise ValueError("Invalid observed source support")
    frames = np.flatnonzero(mask.any(axis=1))
    return np.stack([tokens[i, mask[i]].mean(axis=0, dtype=np.float64) for i in frames]), frames


def margin_parts(memories, positives, bank, weights):
    similarities = np.clip(memories.astype(float) @ positives.astype(float).T, -1., 1.)
    lower = np.quantile(similarities, .1, axis=1)
    tail = cohort_tail(memories, bank, weights)
    return lower - tail, lower, tail


@torch.no_grad()
def adapt(original, codes, predictor, positive_raw, bank, weights, candidates):
    original = np.asarray(original, dtype=np.float64).reshape(768)
    codes = np.asarray(codes, dtype=np.float32).reshape(1024)
    positives = predictor.encode(positive_raw)
    original_memory = predictor.encode(original)[0]
    base, base_positive, base_tail = margin_parts(original_memory[None], positives, bank, weights)
    active = np.flatnonzero(codes != 0)
    effect = np.zeros(1024)
    positive_change = np.zeros(1024)
    negative_change = np.zeros(1024)
    def evaluate(gains):
        raw = residual_edit(torch.from_numpy(original[None]), torch.from_numpy(codes[None]),
            torch.from_numpy(gains), predictor.dictionary, torch.from_numpy(predictor.token_scale)).numpy()
        memory = predictor.encode(raw)
        margin, positive, negative = margin_parts(memory, positives, bank, weights)
        return raw, memory, margin - base[0], positive - base_positive[0], negative - base_tail[0]
    for start in range(0, len(active), 128):
        selected = active[start:start + 128]
        gains = np.zeros((len(selected), 1024), dtype=np.float32)
        gains[np.arange(len(selected)), selected] = -1.
        _, _, effect[selected], positive_change[selected], negative_change[selected] = evaluate(gains)
    order = np.argsort(-effect, kind="stable")
    order = order[(effect[order] > 0) & (codes[order] != 0)]
    gains = np.zeros((len(candidates), 1024), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        gains[i, order[:candidate["k"]]] = -candidate["strength"]
    raw, memories, objective, positive, negative = evaluate(gains)
    if not np.array_equal(raw[0], original):
        raise ValueError("Zero edit changed the source")
    objective[0], positive[0], negative[0] = 0., 0., 0.
    return dict(choice=int(np.argmax(objective)), ordered_components=order, component_effect=effect,
        component_positive_change=positive_change, component_negative_change=negative_change,
        candidates_raw=raw, candidates_memory=memories, candidate_objective=objective,
        candidate_positive_change=positive, candidate_negative_change=negative, gains=gains,
        positive_memory=positives, original_memory=original_memory, pooled_codes=codes)


def fit_view_svm(positives, bank, weights):
    samples = np.vstack([positives, bank]).astype(float)
    count = len(positives)
    labels = np.r_[np.ones(count, dtype=int), -np.ones(len(bank), dtype=int)]
    sample_weight = np.r_[np.full(count, .5 / count), .5 * weights] * (len(bank) + 1)
    model = LinearSVC(C=10., penalty="l2", loss="squared_hinge", dual=False,
        fit_intercept=True, intercept_scaling=1., tol=1e-6, max_iter=10000)
    model.fit(samples, labels, sample_weight=sample_weight)
    if int(model.n_iter_) >= model.max_iter:
        raise ValueError("Observed-view SVM did not converge")
    vector = np.r_[model.coef_[0], model.intercept_[0]]
    if not np.allclose(samples @ vector[:128] + vector[128], model.decision_function(samples), atol=1e-12):
        raise ValueError("SVM coefficient export differs")
    return vector
