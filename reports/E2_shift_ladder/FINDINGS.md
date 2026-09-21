# E2 — Does the physics prior buy robustness, or only accuracy? (RQ1)

**Data:** 480 real VED segments, 134 derived routes, 2 usable vehicles. 5 seeds. Provenance `REAL`.
**Metric:** degradation ratio `D = MAPE(Lk) / MAPE(L1)`. Lower = degrades more gracefully.

| level | shift |
|---|---|
| L0 | random rows (leaky control) |
| L1 | unseen routes (reference) |
| L2 | unseen temperature regime — train warm, test cold |
| L3 | unseen vehicle |
| L3b | unseen vehicle **and** unseen routes (stricter) |

## Why raw D cannot be read directly

Each level scores a **different test set**, so a low D may mean *"easier test set"*
rather than *"robust model"*. The evidence is `rated_range`, a constant predictor
that cannot possibly be robust, yet shows **D = 0.565 at L2** — the cold subset is
simply easier for it.

`physics_only` fits nothing, so its D measures test-set difficulty alone. Dividing by
it isolates degradation incurred **because a model learned**.

## Raw and normalised degradation

| model | D(L2) | D(L3) | D(L3b) | **norm D(L2)** | **norm D(L3b)** |
|---|---|---|---|---|---|
| physics_only | 0.799 | 0.972 | 0.972 | 1.000 | 1.000 |
| ridge | 0.959 | 1.367 | 1.350 | 1.199 | **1.389** |
| hybrid_physics_ml | 0.966 | 1.798 | 1.624 | 1.208 | 1.671 |
| random_forest | 1.026 | 1.646 | 1.644 | 1.284 | 1.692 |
| gbm | 1.009 | 1.674 | 1.733 | 1.262 | 1.783 |

Mean normalised D over the genuine shifts (L2, L3b): **ridge 1.294 < hybrid 1.440 <
random_forest 1.488 < gbm 1.522**, with `physics_only` at 1.000 by construction.

## Against the pre-registered hypotheses

**H1b — "D_hybrid < D_pure-ML at L2 and L4." PARTIALLY SUPPORTED.**
The hybrid degrades less than both tree models (1.440 vs 1.488 and 1.522) but **more
than ridge (1.294)**. The prior helps relative to trees; it does not make the hybrid
the most robust model available.

**H1c — "random_forest and gbm will degrade most sharply at L2." REFUTED.**
Temperature shift barely moves anyone: raw D at L2 is 0.96–1.03 for every learned
model. Training on warm segments and testing on cold costs almost nothing here. A
likely reason is that the dominant cold-weather mechanism, HVAC load, is a
**measured input feature**, so the model does not have to extrapolate to it — it is
told. That is an argument for measuring auxiliary power rather than inferring it.

**H1d — "L0 will overstate accuracy by ≥30 %." REFUTED in magnitude.**
The leaky split flatters models by 8–16 % (E0), not 30 %.

## The result that matters most

**Under vehicle shift, learning stops paying.** Per-fold at L3b:

| model | hold out veh 10 (test 187) | hold out veh 455 (test 283) | veh 541 (test 10) |
|---|---|---|---|
| hybrid_physics_ml | **17.90** | 24.93 | 10.74 |
| physics_only | 26.10 | **20.60** | 10.73 |
| ridge | 22.99 | 28.96 | 22.24 |
| gbm | 21.37 | 38.13 | 23.50 |
| random_forest | 24.06 | 39.21 | 23.39 |

On the **largest fold** (hold out vehicle 455, 283 test segments, only 139 training
segments survive route-disjointness), **`physics_only` at 20.60 % beats every learned
model**, including the hybrid at 24.93 % and `gbm` at 38.13 %.

> **RQ1, answered honestly:** in-distribution the hybrid is clearly best (13.5 % vs
> 23.2 %, E1). Under vehicle shift with little same-vehicle training data, **the
> unlearned physics model wins**. The value of the learned residual is *conditional on
> having training data from a similar vehicle* — it is not a free improvement.

This is the kind of result the degradation ratio was designed to expose and that an
accuracy-only comparison would have hidden completely.

## Limitations — these are severe and bound every claim above

- ⚠️ **n = 2 usable vehicles.** Vehicle 541 contributes 10 segments; its fold is
  reported but carries no weight. **L3/L3b are indicative, not established.**
- **L2 discards ~42 % of segments** to keep routes disjoint, leaving ~38 cold test
  segments per seed. The L2 estimate is noisy.
- The L3b fold for vehicle 455 trains on only 139 segments, so "learning fails under
  vehicle shift" is confounded with "learning fails with little data". **These two
  explanations are not separable here**, and the honest reading is that they compound.
- Cold segments co-occur with high HVAC load and winter driving; L2 is a *seasonal*
  shift, not a clean temperature manipulation.
- All associations, no causation. One vehicle model, one city, one year.

## ⚠️ SUPERSEDED IN PART BY E4

The conclusion above — that `physics_only` wins under vehicle shift — **was an
artefact of data scarcity, not of vehicle shift.** The vehicle-455 fold trained on
only 139 segments, and this document already flagged that the two explanations were
not separable here.

E4 separated them. Using 1,897 PHEV electric-mode segments across 23 vehicles,
leave-one-vehicle-out with ample training data gives the **hybrid 15.99 %** against
**physics_only 26.73 %**. With enough data, learning wins under genuine vehicle shift.

The correct statement is therefore: **the learned residual's value is conditional on
training-data volume, not on vehicle identity.** The text above is left unchanged as
the record of what E2 alone could support.

## What changed in the plan

`physics_only` is promoted from "floor baseline" to **the most shift-robust model
tested**. Any deployment claim about an unseen vehicle should lean on the physics
model, with the ML residual applied only where same-vehicle data exists. That was not
the expected outcome.
