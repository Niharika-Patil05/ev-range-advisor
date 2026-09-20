"""Report figures (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .features.trip_features import TARGET


def make_all_plots(results: dict, out: Path) -> None:
    out = Path(out)
    test, preds = results["test"], results["preds"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    lim = [test[TARGET].min() * 0.9, test[TARGET].max() * 1.1]
    for ax, name in zip(axes, ["rated_range", "physics_only", "hybrid_physics_ml"]):
        ax.scatter(test[TARGET], preds[name], s=8, alpha=0.5)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(title=name, xlabel="actual Wh/km")
    axes[0].set_ylabel("predicted Wh/km")
    fig.tight_layout(); fig.savefig(out / "fig_pred_vs_actual.png", dpi=150); plt.close(fig)

    rr = results["range_results"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(rr.index, rr["MAPE"]); ax.invert_yaxis()
    ax.set(xlabel="holdout MAPE %  (unseen routes)", title="Consumption prediction error")
    fig.tight_layout(); fig.savefig(out / "fig_model_comparison.png", dpi=150); plt.close(fig)

    ab = results["ablation"].dropna(subset=["gbm_CV_MAPE"])
    x = np.arange(len(ab))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, ab["gbm_CV_MAPE"], 0.4, label="pure ML (gbm)")
    ax.bar(x + 0.2, ab["hybrid_CV_MAPE"], 0.4, label="hybrid (physics + ML)")
    ax.set_xticks(x); ax.set_xticklabels([v.replace("without ", "- ") for v in ab["variant"]], rotation=15, ha="right")
    ax.set(ylabel="group-CV MAPE %", title="Ablation"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_ablation.png", dpi=150); plt.close(fig)

    oof = results["soh_oof"]
    oof = oof[oof.model == "random_forest"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(oof["soh"], oof["pred"], s=3, alpha=0.4)
    ax.plot([0.7, 1], [0.7, 1], "k--", lw=1)
    ax.set(xlabel="true SoH", ylabel="predicted SoH", title="SoH, leave-one-battery-out")
    fig.tight_layout(); fig.savefig(out / "fig_soh_lobo.png", dpi=150); plt.close(fig)
