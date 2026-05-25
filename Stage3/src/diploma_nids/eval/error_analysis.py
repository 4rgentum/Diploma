"""Per-class recall and basic FP/FN slicers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_class_recall(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    attack_cat: np.ndarray,
) -> pd.DataFrame:
    """Return a recall-by-attack-category dataframe.

    For ``attack_cat`` rows where ``y_true_binary == 0`` we report FPR — that
    keeps a single output table for both classes.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true_binary).astype(np.int64),
            "y_pred": np.asarray(y_pred_binary).astype(np.int64),
            "attack_cat": np.asarray(attack_cat),
        }
    )
    rows: list[dict[str, float | str | int]] = []
    for cat, g in df.groupby("attack_cat"):
        support = int(len(g))
        if cat == "Normal":
            fp = int(((g["y_true"] == 0) & (g["y_pred"] == 1)).sum())
            metric = fp / support if support > 0 else 0.0
            rows.append({"attack_cat": str(cat), "support": support, "recall_or_fpr": float(metric), "type": "fpr"})
        else:
            tp = int(((g["y_true"] == 1) & (g["y_pred"] == 1)).sum())
            fn = int(((g["y_true"] == 1) & (g["y_pred"] == 0)).sum())
            denom = max(1, tp + fn)
            rows.append({
                "attack_cat": str(cat),
                "support": support,
                "recall_or_fpr": float(tp / denom),
                "type": "recall",
                "tp": tp,
                "fn": fn,
            })
    return pd.DataFrame(rows).sort_values(["type", "attack_cat"]).reset_index(drop=True)
