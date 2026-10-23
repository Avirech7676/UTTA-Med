"""Harmful-adaptation analysis: before vs after vs ground truth."""

from __future__ import annotations

from typing import Dict

import numpy as np


def flip_table(y_true, p_before, p_after, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true).reshape(-1).astype(int)
    b = (np.asarray(p_before).reshape(-1) >= threshold).astype(int)
    a = (np.asarray(p_after).reshape(-1) >= threshold).astype(int)

    correct_before = b == y
    correct_after = a == y

    n_wrong_before = int((~correct_before).sum())
    n_correct_before = int(correct_before.sum())
    n_corr = int((~correct_before & correct_after).sum())  # Wrong → Correct
    n_harm = int((correct_before & ~correct_after).sum())  # Correct → Wrong
    n_stable = int((correct_before & correct_after).sum())
    n_persist = int((~correct_before & ~correct_after).sum())
    n_flip = int((b != a).sum())

    return {
        "n": int(len(y)),
        "correction_count": n_corr,
        "harm_count": n_harm,
        "stable_count": n_stable,
        "persistent_wrong_count": n_persist,
        "flip_count": n_flip,
        "correction_rate": n_corr / max(n_wrong_before, 1),
        "harm_rate": n_harm / max(n_correct_before, 1),
        "flip_rate": n_flip / max(len(y), 1),
        "stable_rate": n_stable / max(len(y), 1),
    }
