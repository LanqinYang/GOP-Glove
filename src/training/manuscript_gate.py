"""The four-case confidence gate used by the DA-LGBM manuscript."""
import numpy as np


def confidence_gated_predict(adann_prob, lgb_prob, adann_threshold=0.5,
                            lgb_threshold=0.5, static_class=10):
    a = np.asarray(adann_prob, dtype=np.float32)
    b = np.asarray(lgb_prob, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 11:
        raise ValueError("Expected matching N by 11 branch probability matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Branch probabilities must be finite")
    ya, yb = a.argmax(axis=1), b.argmax(axis=1)
    ca, cb = a.max(axis=1), b.max(axis=1)
    ma = np.sort(a, axis=1)[:, -1] - np.sort(a, axis=1)[:, -2]
    mb = np.sort(b, axis=1)[:, -1] - np.sort(b, axis=1)[:, -2]
    high_a, high_b = ca >= adann_threshold, cb >= lgb_threshold
    result = np.full(len(a), int(static_class), dtype=int)
    result[high_a & ~high_b] = ya[high_a & ~high_b]
    result[high_b & ~high_a] = yb[high_b & ~high_a]
    both = high_a & high_b
    result[both] = np.where(ya[both] == yb[both], yb[both],
                            np.where(ma[both] > mb[both], ya[both], yb[both]))
    return result


def gated_decision_scores(adann_prob, lgb_prob, **kwargs):
    """One-hot decision encoding for APIs that evaluate labels via argmax."""
    labels = confidence_gated_predict(adann_prob, lgb_prob, **kwargs)
    scores = np.zeros((len(labels), 11), dtype=np.float32)
    scores[np.arange(len(labels)), labels] = 1.0
    return scores
