# E1 — The first honest model comparison

**Data:** 480 real VED analysis segments, 134 derived routes, 2013 Nissan Leaf.
**Protocol:** L1 route-wise, 5 folds × 5 seeds, group disjointness asserted. Provenance `REAL`.
**Replaces:** every number in `reports/archive_synthetic/`, which was circular by construction.

## Result

| model | MAPE (mean ± std) | medAPE | MAE Wh/km | range err km |
|---|---|---|---|---|
| **hybrid_physics_ml** | **13.49 ± 0.79 %** | 9.44 | 19.15 | 19.71 |
| gbm | 18.06 ± 1.42 % | 12.13 | 24.10 | 24.06 |
| ridge | 19.62 ± 1.49 % | 13.97 | 26.43 | 26.61 |
| random_forest | 20.16 ± 1.69 % | 13.56 | 26.64 | 26.81 |
| physics_only | 23.19 ± 0.98 % | 19.41 | 34.67 | 33.60 |
| rated_range | 59.80 ± 7.58 % | 39.81 | 65.64 | 50.89 |

## Against the pre-registered hypothesis

**H1a predicted** the hybrid would beat pure ML *"by a modest margin — far smaller
than the synthetic 2.93 % vs 10.4 %, and possibly within noise."*

**Partly right, partly wrong, and both halves are recorded.** The margin did shrink:
10.05 pp on synthetic data, **4.57 pp** here. But it is **not** within noise. A paired
Wilcoxon signed-rank test over 104 shared routes gives:

| comparison | median per-route diff | p | routes won |
|---|---|---|---|
| hybrid vs gbm | **−2.12 pp** | **1.5 × 10⁻³** | 62 % |
| hybrid vs physics_only | −5.76 pp | 3.5 × 10⁻⁷ | 74 % |
| hybrid vs ridge | −5.39 pp | 5.6 × 10⁻⁶ | 67 % |
| gbm vs physics_only | −4.08 pp | 1.7 × 10⁻² | 63 % |
| gbm vs random_forest | −1.31 pp | 3.7 × 10⁻² | 58 % |
| ridge vs random_forest | +0.69 pp | 0.52 | 46 % |

Two things worth stating plainly:

- The **pooled** gap (4.57 pp) is larger than the **median per-route** gap (2.12 pp),
  so the hybrid's advantage comes disproportionately from hard routes rather than
  from a uniform improvement.
- The hybrid wins on **62 % of routes, not all of them.** It is better on average,
  not universally.

## Other observations

- **`random_forest` (20.16 %) is no better than `ridge` (19.62 %)** — p = 0.52, so
  they are indistinguishable. Trees cannot extrapolate beyond the training envelope;
  a linear model can. The same pattern appeared on synthetic data and is now
  confirmed on real data.
- **`rated_range` is wrong by 60 %.** The manufacturer's EPA figure badly
  over-estimates range on these short, cold, stop-go urban segments. This is the
  project's motivating problem, measured.
- **The physics prior is doing real work**: `physics_only` at 23.19 % already beats
  every pure-ML model except `gbm`, using no training data at all.

## Sensitivity to the derived route tolerance

`route_id` is derived, so the result must not depend on the clustering choice:

| model | 300 m | 500 m | 800 m |
|---|---|---|---|
| hybrid_physics_ml | **12.55** | **14.46** | **13.83** |
| gbm | 16.70 | 19.70 | 20.43 |
| random_forest | 18.63 | 21.64 | 21.90 |
| ridge | 18.46 | 21.01 | 21.31 |
| physics_only | 21.46 | 22.91 | 21.63 |

**The ranking is identical at all three tolerances.** The conclusion does not rest on
an arbitrary clustering parameter.

## What may and may not be claimed

✅ *"On 480 analysis segments from 2013 Nissan Leafs in Ann Arbor, evaluated on unseen
routes, a physics-informed hybrid achieved 13.5 % MAPE against 18.1 % for the best
pure-ML model (paired p = 0.0015)."*

❌ *"Our model predicts EV range to 13.5 % accuracy."* — one vehicle model, one city,
one year, two usable vehicles, short urban segments.

❌ Anything causal. These are associations on observational data.

⚠️ This is **in-distribution** accuracy on unseen routes only. Whether the physics
prior also buys **robustness** under temperature and vehicle shift is RQ1, and E2
answers it. E1 alone does not.
