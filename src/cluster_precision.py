"""Small auditable calculations for geometry-cluster evidence limits."""

from __future__ import annotations


def all_successes_clopper_pearson_lower(
    cluster_count: int, confidence: float = 0.95
) -> float:
    """Two-sided exact lower bound after success in every independent cluster.

    For ``n`` successes out of ``n``, the Clopper--Pearson lower endpoint is
    ``((1 - confidence) / 2) ** (1 / n)``. This is only a precision illustration
    for a Bernoulli sign-consistency estimand; it is not an effect-size power
    calculation and does not turn nested videos or frames into independent data.
    """

    if not isinstance(cluster_count, int) or isinstance(cluster_count, bool):
        raise ValueError("cluster_count must be an integer")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be numeric")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = 1.0 - confidence
    return (alpha / 2.0) ** (1.0 / cluster_count)
