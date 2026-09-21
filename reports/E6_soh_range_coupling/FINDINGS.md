# E6 — How much of the range error is the battery's fault? (RQ3)

> ## ⚠️ COUPLED-SIM — a propagation study, **not** end-to-end validation
> No public dataset contains both a vehicle's trips and its own pack's SoH trajectory.
> The two halves are joined **only** through the physical identity
> `range = usable_energy(SoH, SoC, T) / consumption`, never by merging rows.
> The consumption error is **real** (VED, E1). The SoH error is **real** (NASA, E5).
> **The coupling between them is simulated**, because a lab 2 Ah 18650 is not the
> 24 kWh pack in that Leaf. Nothing here shows the system works end to end.

## Inputs — both measured out-of-fold

| source | n | sd(log) | MAE |
|---|---|---|---|
| SoH, leave-one-battery-out (E5) | 2,359 | 0.1179 | 6.57 % |
| consumption, unseen routes (E1) | 491 | 0.1952 | 14.46 % |

## Result 1 — the consumption model dominates, not the battery

| source | sd(log) | share of variance |
|---|---|---|
| **consumption model** | 0.1952 | **71.9 %** |
| SoH estimation | 0.1179 | 26.2 % |
| temperature derate | 0.0320 | 1.9 % |
| **total** | **0.2303** | ≈ **23 % relative range uncertainty (1σ)** |

**H3c — "at long trips and low SoC, SoH error will contribute more variance than
consumption error." REFUTED.** Consumption error contributes nearly three times as
much.

And the reason matters more than the verdict. **The shares do not depend on trip
length or starting SoC at all.** Both errors are *multiplicative*: in logs the
identity separates into independent additive terms, so the variance shares are
scale-free. The hypothesis assumed consumption error would average out over a longer
trip — it does not, because it is a relative error on a per-kilometre basis, not an
accumulating random walk. **The premise was wrong, not just the prediction.**

## Result 2 — the engineering deliverable

**km of range error per percentage point of SoH error:**

| SoH | SoC | range (km) | mean km error | **km per pp** |
|---|---|---|---|---|
| 1.00 | 100 % | 164.1 | 8.72 | **1.628** |
| 0.90 | 100 % | 147.7 | 8.04 | **1.630** |
| 0.80 | 100 % | 131.3 | 7.34 | **1.632** |
| 1.00 | 50 % | 82.1 | 4.33 | 0.812 |
| 0.90 | 50 % | 73.9 | 4.01 | 0.815 |
| 0.80 | 50 % | 65.7 | 3.69 | 0.816 |

**H3d — "the relationship will be approximately linear across the 80–100 % SoH band,
giving a stable km-per-pp figure." SUPPORTED.** 1.628 / 1.630 / 1.632 — stable to
three decimal places, and it halves with SoC exactly as the identity requires.

> **≈ 1.6 km of range error per percentage point of SoH error at full charge,
> ≈ 0.8 km at half charge**, for a 24 kWh Leaf on Ann Arbor-like driving.

**H3a — "real LOBO SoH error will be 2–5 pp, far worse than the synthetic 1.11 pp."
SUPPORTED.** E5 measured **4.68 pp**, at the top of the predicted range and 4.2× the
circular synthetic figure.

## What this means for the project

1. **Improving the consumption model matters ~3× more than improving SoH estimation**
   for range accuracy. That is the opposite of where a battery-focused project would
   naturally put its effort, and it is the most actionable thing E6 produced.
2. **The temperature derate contributes almost nothing** (1.9 %), even when its
   placeholder coefficient is swept across a 3× range. Calibrating it further is not
   worth the effort — a useful negative for the plan's E8.
3. The advisor can legitimately state "±1.6 km per point of battery health" in the
   UI, **provided it carries the COUPLED-SIM caveat**.

## Limitations — these are not minor

- **Lab cells are not vehicle packs.** NASA cells are 2 Ah 18650s cycled in a
  controlled rig; a pack adds cell imbalance, BMS logic and thermal gradients. The
  assumption that their error distribution transfers is the single largest leap here.
- **The SoH error has a heavy tail.** E5's worst cell was 13.9 pp, so the mean
  understates the risk for an unlucky pack.
- **Error sources are assumed independent.** They are not perfectly so — a cold day
  raises both consumption error and SoH estimation error.
- **Nominal pack energy is held at 24 kWh.** E7 separately measured an *effective*
  usable pack energy of 14.5–15.7 kWh for these two vehicles, which would shift the
  absolute kilometres though not the per-pp slope.
- One vehicle model, one city, one year.
