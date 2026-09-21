"""RQ2 - is the range band trustworthy WHERE IT MATTERS?

Split conformal guarantees MARGINAL coverage: 90% on average. That is the wrong
guarantee for range anxiety, because a rider is stranded by a band that fails on a
cold evening, not by one that fails on average.

Four interval methods, compared on marginal AND per-subgroup coverage:

    naive quantile     the existing method -- included as the negative control,
                       since it measured 0.73 against a 0.90 target on synthetic data
    split conformal    the existing fix; marginal guarantee only
    CQR                conformalised quantile regression: adaptive width AND a guarantee
    Mondrian           group-conditional conformal: a separate quantile per subgroup

Subgroups are temperature tertiles, fixed in advance in docs/HYPOTHESES.md so they
cannot be chosen after seeing which split flatters the method.

Run:  python -m src.experiments.e_rq2_uncertainty
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import NISSAN_LEAF_2013, REPORTS_DIR  # noqa: E402
from src.evaluation.harness import run_experiment  # noqa: E402
from src.evaluation.leakage_checks import assert_three_way_disjoint  # noqa: E402
from src.evaluation.provenance import ConclusionType, Evidence, Provenance  # noqa: E402
from src.experiments._common import load_segments  # noqa: E402
from src.features.segment_features import ALL_FEATURES, TARGET  # noqa: E402
from src.models.range_model import HybridResidual  # noqa: E402
from src.models.uncertainty import (ConformalInterval, ConformalizedQuantile,  # noqa: E402
                                    MondrianConformal, QuantileHybrid, coverage,
                                    coverage_by_group)

ALPHA = 0.10


def temperature_group(df: pd.DataFrame, edges: tuple[float, float]) -> np.ndarray:
    lo, hi = edges
    return np.where(df.temp_c <= lo, "cold", np.where(df.temp_c <= hi, "mild", "warm"))


def three_way_split(df: pd.DataFrame, seed: int):
    """Route-disjoint train / calibration / test, 50 / 25 / 25.

    Calibration MUST share no route with train (optimistically small residuals give
    a band that is too narrow) or with test (which invalidates the guarantee).
    """
    rng = np.random.default_rng(seed)
    routes = pd.unique(df.route_id)
    rng.shuffle(routes)
    n = len(routes)
    tr_r, cal_r, te_r = routes[:n // 2], routes[n // 2:3 * n // 4], routes[3 * n // 4:]
    tr = df[df.route_id.isin(tr_r)].reset_index(drop=True)
    cal = df[df.route_id.isin(cal_r)].reset_index(drop=True)
    te = df[df.route_id.isin(te_r)].reset_index(drop=True)
    assert_three_way_disjoint(tr, cal, te, "route_id")
    return tr, cal, te


def evaluate_methods(tr, cal, te, edges, seed) -> pd.DataFrame:
    hyb = HybridResidual(ALL_FEATURES, seed=seed).fit(tr, tr[TARGET])
    p_cal, p_te = hyb.predict(cal), hyb.predict(te)
    g_cal, g_te = temperature_group(cal, edges), temperature_group(te, edges)
    y_te = te[TARGET].to_numpy(float)

    bands = {}
    qh = QuantileHybrid(ALL_FEATURES, ALPHA, seed).fit(tr, tr[TARGET])
    bands["naive_quantile"] = qh.interval(te)
    bands["split_conformal"] = ConformalInterval(ALPHA).fit(cal[TARGET], p_cal).interval(p_te)
    cqr = ConformalizedQuantile(ALL_FEATURES, ALPHA, seed).fit(tr, tr[TARGET]).calibrate(cal, cal[TARGET])
    bands["cqr"] = cqr.interval(te)
    mond = MondrianConformal(ALPHA).fit(cal[TARGET], p_cal, g_cal)
    bands["mondrian"] = mond.interval(p_te, g_te)

    rows = []
    for name, (lo, hi) in bands.items():
        by = coverage_by_group(y_te, lo, hi, g_te).set_index("group")
        mid = 0.5 * (np.asarray(lo) + np.asarray(hi))
        rec = dict(method=name,
                   marginal_coverage=coverage(y_te, lo, hi),
                   mean_rel_width=float(np.mean((np.asarray(hi) - np.asarray(lo)) / mid)),
                   worst_group_coverage=float(by.coverage.min()),
                   worst_group=str(by.coverage.idxmin()), n_test=len(te))
        for g in ("cold", "mild", "warm"):
            rec[f"cov_{g}"] = float(by.coverage.get(g, np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    seg = load_segments()
    edges = tuple(seg.temp_c.quantile([1 / 3, 2 / 3]))
    print(f"RQ2: {len(seg)} BEV segments, {seg.route_id.nunique()} routes")
    print(f"temperature tertiles: cold <= {edges[0]:.1f} C < mild <= {edges[1]:.1f} C < warm\n")

    res = run_experiment(
        "E_RQ2_uncertainty",
        "Marginal vs conditional coverage of four interval methods",
        lambda seed: evaluate_methods(*three_way_split(seg, seed), edges, seed),
        Evidence(
            experiment="RQ2 - conditional coverage of the range band",
            measured=["HV battery current and voltage", "onboard air temperature",
                      "vehicle speed", "HVAC power"],
            derived=["wh_per_km", "hybrid point prediction", "temperature tertiles"],
            external=["eVED elevation, gradient, speed limits"],
            assumptions=["alpha = 0.10", "tertile edges fixed in advance (HYPOTHESES.md)",
                         "conformal assumes exchangeability, which shift violates"],
            split_protocol="route-disjoint train/calibration/test 50/25/25, 5 seeds",
            n=f"{len(seg)} segments, {seg.route_id.nunique()} routes",
            provenance=Provenance.REAL,
            conclusion_type=ConclusionType.DEMONSTRATED,
            claim_strength="coverage measured on held-out routes for one vehicle model",
            known_confounds=["cold segments co-occur with high HVAC load",
                             "calibration sets are small (~120 segments), so group "
                             "quantiles rest on few points"],
        ),
        REPORTS_DIR, index_cols=["method"], verbose=True,
        config=dict(alpha=ALPHA, tertile_edges=list(edges)),
    )

    s = res["summary"].set_index("method")
    print(f"\n=== coverage at target {1-ALPHA:.2f} (mean over 5 seeds) ===")
    print(f"{'method':18s}{'marginal':>10}{'cold':>8}{'mild':>8}{'warm':>8}"
          f"{'worst':>8}{'width':>8}")
    for m in ("naive_quantile", "split_conformal", "cqr", "mondrian"):
        r = s.loc[m]
        print(f"{m:18s}{r['marginal_coverage_mean']:10.3f}{r['cov_cold_mean']:8.3f}"
              f"{r['cov_mild_mean']:8.3f}{r['cov_warm_mean']:8.3f}"
              f"{r['worst_group_coverage_mean']:8.3f}{r['mean_rel_width_mean']:8.3f}")
    print("\n  'width' is the mean interval width relative to the prediction.")
    print("  Coverage without width is meaningless: a band of +/-inf covers everything.")

    # ---- H2e: coverage UNDER SHIFT -------------------------------------------------
    # Conformal prediction guarantees coverage under EXCHANGEABILITY. Distribution
    # shift violates that by construction, so the guarantee should not be expected to
    # survive. Here train and calibrate on warm segments, then test on cold ones.
    print("\n=== H2e: calibrate on WARM, test on COLD (exchangeability violated) ===")
    rows = []
    for seed in (0, 1, 2, 3, 4):
        warm = seg[seg.temp_c > edges[0]].reset_index(drop=True)
        cold = seg[seg.temp_c <= edges[0]].reset_index(drop=True)
        # keep routes disjoint between the warm calibration set and the cold test set
        rng = np.random.default_rng(seed)
        wr = pd.unique(warm.route_id); rng.shuffle(wr)
        tr = warm[warm.route_id.isin(wr[:len(wr) * 2 // 3])].reset_index(drop=True)
        cal = warm[warm.route_id.isin(wr[len(wr) * 2 // 3:])].reset_index(drop=True)
        te = cold[~cold.route_id.isin(set(tr.route_id) | set(cal.route_id))].reset_index(drop=True)
        if len(te) < 20 or len(cal) < 30:
            continue
        hyb = HybridResidual(ALL_FEATURES, seed=seed).fit(tr, tr[TARGET])
        p_cal, p_te = hyb.predict(cal), hyb.predict(te)
        y_te = te[TARGET].to_numpy(float)
        lo, hi = ConformalInterval(ALPHA).fit(cal[TARGET], p_cal).interval(p_te)
        cqr = ConformalizedQuantile(ALL_FEATURES, ALPHA, seed).fit(tr, tr[TARGET]).calibrate(cal, cal[TARGET])
        clo, chi = cqr.interval(te)
        rows.append(dict(seed=seed, n_cal=len(cal), n_test=len(te),
                         split_conformal=coverage(y_te, lo, hi),
                         cqr=coverage(y_te, clo, chi)))
    if rows:
        sh = pd.DataFrame(rows)
        print(f"  calibrated on {sh.n_cal.mean():.0f} warm segments, tested on "
              f"{sh.n_test.mean():.0f} cold ones, {len(sh)} seeds")
        for m in ("split_conformal", "cqr"):
            print(f"    {m:18s} coverage {sh[m].mean():.3f} +/- {sh[m].std():.3f}  "
                  f"(target {1-ALPHA:.2f})")
        sh.to_csv(res["out_dir"] / "coverage_under_shift.csv", index=False)
    else:
        print("  not enough cold segments on unseen routes to run the shifted test")


if __name__ == "__main__":
    main()
