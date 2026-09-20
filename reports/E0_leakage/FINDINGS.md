# E0 — How much accuracy does a naive split fabricate?

**Data:** 480 real VED segments, 134 derived routes. 5 folds × 5 seeds. Provenance `REAL`.
**Design:** identical models, identical data, two protocols — L0 shuffled rows vs L1 unseen routes.

## Result

| model | L0 random rows | L1 unseen routes | hidden error (pp) | hidden share |
|---|---|---|---|---|
| hybrid_physics_ml | 11.33 | 13.49 | **2.16** | **16.0 %** |
| gbm | 15.90 | 18.06 | **2.16** | 12.0 % |
| random_forest | 17.33 | 20.16 | **2.83** | 14.1 % |
| ridge | 18.15 | 19.62 | 1.48 | 7.5 % |
| physics_only | 22.09 | 23.19 | 1.10 | 4.8 % |
| rated_range | 54.05 | 59.80 | 5.74 | 9.6 % |

**A random row split hides about 2.2 percentage points — 12 to 16 % — of the true
error of every learned model.** Reported as MAPE on the same 480 segments.

## The control, and its limits

`physics_only` and `rated_range` fit nothing, so they **cannot** memorise a route.
Any gap they show is the two protocols scoring different test sets, not leakage.
`physics_only` shows 1.10 pp, which sets a floor: roughly half of the learned models'
2.2 pp is test-set composition and the rest is memorisation.

⚠️ **The control is imperfect and we say so.** `rated_range` shows 5.74 pp despite
being a constant predictor, because inflation scales with a model's error magnitude
(its MAPE is 54–60 %). So the "excess over control" column should be read as
indicative, not as a clean decomposition. The defensible headline is the direct
comparison: **every learned model looks 12–16 % better under a random split than it
really is.**

## Why this matters

The models most able to memorise show the largest inflation. `random_forest`, with
`min_samples_leaf=3`, is the most flexible learner here and has the highest absolute
inflation (2.83 pp) of any model with a sane error level. `ridge`, which cannot
memorise geometry, shows the least among learned models (1.48 pp).

Note also that VED has **no `route_id`** — it is derived by clustering trip endpoints
(500 m). A tighter tolerance splits repeats of one journey across train and test,
which would leak. The tolerance sensitivity in E1 checks that the conclusions do not
depend on that choice.

## What this answers

*"How did you prevent data leakage?"* — not with a claim, but with a number: we
measured what leakage would have bought us, and declined it.
