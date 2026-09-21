# RQ2 — Is the range band trustworthy *where it matters*?

**Data:** 480 real BEV segments, 134 routes. Route-disjoint train/calibration/test 50/25/25, 5 seeds.
**Target coverage:** 90 % (α = 0.10). Subgroups = temperature tertiles, **fixed in advance** in `docs/HYPOTHESES.md`.
Provenance `REAL`.

## Result

| method | marginal | cold | mild | warm | **worst group** | width |
|---|---|---|---|---|---|---|
| naive_quantile | 0.687 | 0.668 | 0.663 | 0.729 | **0.580** | 0.380 |
| split_conformal | 0.948 | 0.978 | 0.911 | 0.948 | 0.898 | 0.684 |
| cqr | 0.935 | 0.913 | 0.924 | 0.971 | 0.889 | 0.680 |
| **mondrian** | 0.953 | 0.942 | 0.955 | 0.964 | **0.931** | 0.745 |

Width is the mean interval width relative to the prediction. **Coverage without width
is meaningless** — a band of ±∞ covers everything — so they are always reported together.

## Against the pre-registered hypotheses

**H2c — "naive quantile will under-cover; CQR restores coverage at the cost of wider
intervals." SUPPORTED, emphatically.**
The existing quantile method covers **0.687** against a 0.90 target — *worse* than the
0.73 measured on synthetic data. CQR restores 0.935, and the cost is exactly as
predicted: **width rises from 0.380 to 0.680**, nearly double. The band was not
mysteriously broken; it was simply never conformalised.

**H2d — "Mondrian will improve worst-subgroup coverage, at the cost of a wider mean
interval." SUPPORTED.**
Worst-group coverage rises from 0.898 to **0.931**, and mean width rises from 0.684 to
0.745. Precisely the predicted trade.

**H2b — "conditional coverage will fail in the coldest subgroup, below 85 %."
REFUTED.**
Split conformal covers **0.978 in the cold** — the *highest* of the three groups, not
the lowest. There is no cold-weather coverage failure to repair.

**H2e — "under shift, all methods will lose coverage." REFUTED.**
Calibrating on ~86 warm segments and testing on ~34 cold ones on unseen routes:

| method | coverage under shift |
|---|---|
| split_conformal | **0.929 ± 0.061** |
| cqr | 0.882 ± 0.110 |

Coverage **survives** a deliberate violation of exchangeability. CQR is the shakier of
the two — higher variance and slipping just under target — so split conformal is the
more robust choice under shift even though CQR adapts better in-distribution.

**H2a — "split conformal will achieve marginal coverage close to 90 %." PARTIALLY.**
It achieves 0.948 — **over**-coverage. With only ~120 calibration segments the
finite-sample correction `⌈(n+1)(1−α)⌉/n` is conservative, so the band is wider than
strictly needed. Safe, but inefficient.

## The unifying explanation

Three refuted hypotheses — **H1c** (trees degrade most under temperature shift),
**H2b** (cold-weather coverage fails), **H2e** (coverage collapses under shift) — all
predicted cold-weather failure. **None of it happened**, and the same mechanism
explains all three:

> **Auxiliary power is a measured input, not something the model must extrapolate to.**
> HVAC is the dominant cold-weather energy mechanism, and because VED instruments it
> directly (100 % availability on the BEVs, median 390 W, r = +0.66 with consumption),
> the model is *told* the main reason cold trips cost more. It never has to guess.

That is a concrete, transferable engineering conclusion: **instrument the auxiliary
load and the cold-weather robustness problem largely disappears.** It also predicts
that a fleet *without* HVAC instrumentation would show the failures these hypotheses
expected — and E4 supplies exactly that case, where PHEV→BEV transfer fails partly
because auxiliary load is only 12 % instrumented on PHEVs.

## Practical recommendation

- **Replace the naive quantile band immediately.** At 0.687 coverage it is unsafe: it
  claims 90 % and delivers 69 %.
- **Use Mondrian conformal** where worst-group safety matters. It costs ~9 % extra
  width for +3.3 points of worst-group coverage.
- **Use split conformal under known shift** — it held 0.929 where CQR fell to 0.882.

## Limitations

- Calibration sets are ~120 segments, so group quantiles rest on few points; the
  shifted test calibrates on ~86 and tests on ~34.
- Over-coverage at these sample sizes is expected and is not evidence of a better method.
- Conformal guarantees are **marginal over the calibration draw**; a single observed
  coverage number is itself an estimate with sampling error.
- One vehicle model, one city, one year. Two usable vehicles.
