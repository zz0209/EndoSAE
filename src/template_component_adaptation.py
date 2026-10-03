import numpy as np
from scipy.special import expit
from sklearn.svm import LinearSVC
import torch

from src.conditional_component_policy import raw_key, weighted_quantiles
from src.token_memory_edit import residual_edit


def cohort_tail(memories, bank, weights):
    scores = np.clip(memories.astype(float) @ bank.astype(float).T, -1., 1.)
    return np.array([weighted_quantiles(row, weights, [.99])[0] for row in scores])


@torch.no_grad()
def adapt_source(original, codes, predictor, bank, weights, candidates):
    original = np.asarray(original, dtype=np.float64).reshape(768)
    codes = np.asarray(codes, dtype=np.float32).reshape(1024)
    if not np.isfinite(codes).all() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid source adaptation inputs")
    base = predictor.encode(original)[0]
    active = np.flatnonzero(codes != 0)
    before = float(cohort_tail(base[None], bank, weights)[0])
    component_benefit = np.zeros(1024)
    component_harm = np.zeros(1024)
    for start in range(0, len(active), 128):
        selected = active[start:start + 128]
        gains = np.zeros((len(selected), 1024), dtype=np.float32)
        gains[np.arange(len(selected)), selected] = -1.
        edited = residual_edit(torch.from_numpy(original[None]), torch.from_numpy(codes[None]),
            torch.from_numpy(gains), predictor.dictionary, torch.from_numpy(predictor.token_scale)).numpy()
        memory = predictor.encode(edited)
        component_benefit[selected] = before - cohort_tail(memory, bank, weights)
        component_harm[selected] = 1 - np.clip(memory.astype(float) @ base.astype(float), -1., 1.)
    objective = component_benefit - component_harm
    order = np.argsort(-objective, kind="stable")
    order = order[(objective[order] > 0) & (component_benefit[order] > 0) & (codes[order] != 0)]
    gains = np.zeros((len(candidates), 1024), dtype=np.float32)
    for i, candidate in enumerate(candidates):
        gains[i, order[:candidate["k"]]] = -candidate["strength"]
    raw = residual_edit(torch.from_numpy(original[None]), torch.from_numpy(codes[None]),
        torch.from_numpy(gains), predictor.dictionary, torch.from_numpy(predictor.token_scale)).numpy()
    if not np.array_equal(raw[0], original):
        raise ValueError("Zero intervention changed the source")
    memories = predictor.encode(raw)
    benefits = before - cohort_tail(memories, bank, weights)
    harms = 1 - np.clip(memories.astype(float) @ base.astype(float), -1., 1.)
    benefits[0], harms[0] = 0., 0.
    joint = benefits - harms
    choice = int(np.argmax(joint))
    return dict(choice=choice, ordered_components=order, component_benefit=component_benefit,
        component_harm=component_harm, candidates_raw=raw, candidates_memory=memories,
        candidate_benefit=benefits, candidate_harm=harms, candidate_objective=joint, gains=gains,
        original_memory=base, active_components=active, pooled_codes=codes)


def fit_template_svm(source, bank, weights):
    samples = np.vstack([source, bank]).astype(np.float64)
    labels = np.r_[1, -np.ones(len(bank), dtype=int)]
    sample_weight = np.r_[.5, .5 * weights] * len(samples)
    model = LinearSVC(C=10., penalty="l2", loss="squared_hinge", dual=False,
                      fit_intercept=True, intercept_scaling=1., tol=1e-6, max_iter=10000)
    model.fit(samples, labels, sample_weight=sample_weight)
    if int(model.n_iter_) >= model.max_iter:
        raise ValueError("Template SVM did not converge")
    exported = np.r_[model.coef_[0], model.intercept_[0]]
    if not np.allclose(samples @ exported[:128] + exported[128], model.decision_function(samples), atol=1e-12):
        raise ValueError("Template SVM export changed predictions")
    return exported, dict(iterations=int(model.n_iter_), positive_templates=1, negative_templates=len(bank),
        class_weighting="Equal total positive and negative mass; equal procedure/lesion mass within negatives",
        C=10., loss="squared_hinge", solver="liblinear primal", coefficient_norm=float(np.linalg.norm(exported[:128])))


class TemplatePredictor:
    def __init__(self, reference, memories, svm=False):
        self.reference, self.memories, self.svm = reference, memories, svm

    def encode(self, raw, batch_size=512):
        raw = np.asarray(raw).reshape(-1, 768)
        return np.concatenate([self.reference(raw[i:i + batch_size])["supcon_l2"] for i in range(0, len(raw), batch_size)])

    def memory(self, raw):
        saved, memory = self.memories[raw_key(raw)]
        if not np.array_equal(np.asarray(raw).reshape(-1), saved):
            raise ValueError("Template identity changed")
        return memory.copy()

    def score_encoded(self, memory, codes):
        score = np.asarray(codes).astype(float) @ memory[:128].astype(float)
        return expit(score + memory[128]) if self.svm else np.clip(score, -1., 1.)

    def score(self, raw, queries, batch_size=512):
        return self.score_encoded(self.memory(raw), self.encode(queries, batch_size))
