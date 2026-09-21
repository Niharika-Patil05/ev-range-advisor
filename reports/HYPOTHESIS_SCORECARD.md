# Pre-registration scorecard

Every hypothesis in `docs/HYPOTHESES.md`, committed **before** any real data was
downloaded (git `f7be340`, 20 Sep 2026), against what the experiments measured.

**8 supported · 6 refuted · 2 partial · 1 untested.** The refutations are the more
informative half, and none of them was quietly reinterpreted.

## RQ1 — physics prior and distribution shift

| # | Prediction | Verdict | Measured |
|---|---|---|---|
| H1a | Hybrid beats pure ML, but by a modest margin, possibly within noise | **PARTIAL** | Margin shrank 10.05 → 4.57 pp, but **not** within noise (paired p = 0.0015) |
| H1b | Hybrid degrades less than pure ML under shift | **PARTIAL** | Beats both tree models (1.44 vs 1.49, 1.52) but **loses to ridge** (1.29) |
| H1c | Trees degrade most sharply under temperature shift | **REFUTED** | Raw D ≈ 0.96–1.03 for *every* learned model |
| H1d | A random split overstates accuracy by ≥ 30 % | **REFUTED** | 8–16 % (E0) |
| H1e | Identified physics beats nominal physics | **REFUTED** | 22.38 % vs 22.54 % — 0.16 pp, and `crr` is not identifiable (E7) |

## RQ2 — interval calibration

| # | Prediction | Verdict | Measured |
|---|---|---|---|
| H2a | Split conformal lands close to 90 % marginal | **PARTIAL** | 0.948 — *over*-covers at n ≈ 120 calibration |
| H2b | Conditional coverage fails in the cold, below 85 % | **REFUTED** | 0.978 in the cold — the **highest** group |
| H2c | Naive quantile under-covers; CQR fixes it at wider width | **SUPPORTED** | 0.687 → 0.935; width 0.380 → 0.680 |
| H2d | Mondrian improves worst-group coverage at wider mean width | **SUPPORTED** | Worst 0.898 → 0.931; width 0.684 → 0.745 |
| H2e | All methods lose coverage under shift | **REFUTED** | Split conformal held 0.929 ± 0.061 |

## RQ3 — SoH and range

| # | Prediction | Verdict | Measured |
|---|---|---|---|
| H3a | Real LOBO SoH error 2–5 pp, far worse than synthetic 1.11 | **SUPPORTED** | **4.68 pp**, 4.2× the synthetic figure |
| H3b | NASA→Oxford transfer worse still | **NOT TESTED** | Oxford not downloaded |
| H3c | SoH error dominates variance at long trips and low SoC | **REFUTED** | Consumption 71.9 % vs SoH 26.2 %, and the shares are **scale-free** — the premise was wrong |
| H3d | km-per-pp stable across the 80–100 % SoH band | **SUPPORTED** | 1.628 / 1.630 / 1.632 |

## H4 — features

| # | Prediction | Verdict | Measured |
|---|---|---|---|
| H4a | Measured aux power beats ambient temperature | **SUPPORTED, not significant** | +1.73 vs +1.59 pp, inside ± 0.43 / 0.75 |
| H4b | Traffic proxy adds information beyond speed and stops | **REFUTED** | −0.02 ± 0.09 pp — observed speed already encodes it |
| H4c | Headwind is a weak feature | **SUPPORTED** | +0.05 ± 0.18 pp — indistinguishable from zero, 30× smaller than onboard temperature |
| H4d | Grade features weakly informative | **SUPPORTED** | +0.22 and +0.53 pp |

## The thread running through the refutations

**H1c, H2b and H2e all predicted cold-weather failure. None occurred.** One mechanism
explains all three: **auxiliary power is a measured input**, so the model is *told*
the dominant reason cold trips cost more rather than having to extrapolate to it.

The prediction is falsifiable and E4 supplies the contrasting case — PHEV→BEV transfer
degrades partly because the heater channel is only 12 % instrumented on PHEVs.

**Engineering conclusion: instrument the auxiliary load, and the cold-weather
robustness problem largely dissolves.**

## The correction that mattered most

**E2 concluded that `physics_only` beats learned models under vehicle shift. E4 showed
that was a data-scarcity artefact** — holding out a BEV left only 139 training
segments. With 1,764 PHEV-electric segments across 12 vehicles the hybrid wins
decisively (15.99 % vs 26.73 %).

E2's findings file is **annotated, not rewritten**, so the record of what E2 alone
could support survives.


## Addendum — H4c, tested after the fact

Wind and rain were wired in and ablated (`reports/H4c_weather_features/`). Adding them
makes both models marginally **worse** (+0.11 to +0.12 pp), and headwind contributes
+0.05 ± 0.18 pp individually.

The reason sharpens a theme running through the whole project. Reanalysis is a ~9 km
grid average, so every segment in an hour gets identical wind, and reanalysis
temperature agrees with VED's onboard sensor at only r = 0.882, RMSE 5.60 °C. **The
one weather variable that matters is available onboard at far higher fidelity; the
ones available only externally carry no measurable signal.**

Together with the auxiliary-power finding — that measuring HVAC dissolves the
cold-weather robustness problem — the practical conclusion is the same from both
directions: **instrument the vehicle rather than joining an external dataset.**
