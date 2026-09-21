# E3 — Which feature groups actually carry signal?

**Data:** 480 real BEV segments, 134 routes, L1 route-wise, 5 folds × 5 seeds. Provenance `REAL`.
**Why it now means something:** on synthetic data the generator built consumption *from*
rain and temperature, so "weather matters" was true by construction. On real data the
answer was unknown in advance.

This is the evidence for **synopsis objective 3** — the impact of weather, traffic and
driving behaviour.

## Group-level ablation

| removed group | gbm ΔMAPE | hybrid ΔMAPE |
|---|---|---|
| **route** | **+7.39 pp (+40.9 %)** | **+7.19 pp (+53.3 %)** |
| state | +3.56 pp (+19.7 %) | −0.03 pp (−0.2 %) |
| weather | +1.46 pp (+8.1 %) | +1.54 pp (+11.4 %) |
| traffic | −0.06 pp (−0.3 %) | −0.09 pp (−0.6 %) |

Baselines: gbm 18.06 %, hybrid 13.49 %.

**The two models differ, and the reason is structural.** `gbm` sees only the listed
features, so dropping a group removes that information entirely. The hybrid keeps
`phys_wh_km`, and the physics layer consumes speed, grade and measured HVAC
*internally* — so dropping the `state` group from the ML layer costs the hybrid
nothing (−0.03 pp) while costing `gbm` 3.56 pp. **The information is still reaching
the hybrid through the physics term.** This asymmetry is a property of the
architecture, not a result about the world.

## Single-feature drops (gbm, 3 seeds)

| dropped feature | ΔMAPE | ± std |
|---|---|---|
| **aux_power_w** (measured HVAC) | **+1.73** | 0.43 |
| **temp_c** (measured onboard) | **+1.59** | 0.75 |
| accel_pos_mean | +0.93 | 0.15 |
| mean_speed_kmh | +0.73 | 0.23 |
| elev_gain_m | +0.53 | 0.25 |
| soc_start | +0.32 | 0.15 |
| stops_per_km | +0.24 | 0.28 |
| abs_grade_pct | +0.22 | 0.27 |
| speed_limit_kmh | +0.20 | 0.27 |
| **speed_deficit_kmh** (traffic proxy) | **−0.02** | 0.09 |

⚠️ Single-feature drops **understate** correlated features: a dropped feature's
information can survive through its neighbours. They are a lower bound on importance,
not a ranking of physical significance.

## Against the pre-registered hypotheses

**H4a — "measured auxiliary power will be a stronger predictor than ambient
temperature alone." DIRECTIONALLY SUPPORTED, NOT SIGNIFICANT.**
aux_power_w (+1.73) does edge out temp_c (+1.59), but the 0.14 pp margin sits well
inside the standard deviations (0.43 and 0.75). **The two are not distinguishable
here.** The defensible claim is that both are first-order and neither dominates.

**H4b — "the traffic proxy will carry information beyond mean speed and stop rate."
REFUTED.**
`speed_deficit_kmh` contributes **−0.02 ± 0.09 pp** individually and the whole traffic
group contributes −0.06 pp. It adds nothing.

The reason is straightforward in hindsight: the proxy is `posted limit − observed
speed`, and **observed speed is already a feature**. Congestion is encoded in how
fast the vehicle actually went; the posted limit adds little on top.

> This is a **useful negative result for the synopsis's traffic objective**. Traffic
> was addressed with a reproducible derived feature rather than a paid API, and the
> honest finding is that once observed speed is available, an explicit congestion
> proxy is redundant. That is a better answer than claiming a traffic feed would have
> helped.

**H4c — "headwind will be a weak feature." NOT TESTED.**
Wind and precipitation from the Open-Meteo archive were **never wired into the
feature table**, even though Gate G1 validated the join. The `weather` group is
currently onboard temperature alone. This is an open gap, not a finding.

**H4d — "grade features will be weakly informative." SUPPORTED.**
`abs_grade_pct` +0.22 and `elev_gain_m` +0.53 are both small, consistent with the
terrain analysis: 96.7 % of rows have |gradient| < 0.5 %, and the measured grade
attribution is a median 0.3 Wh per segment.

## What this licenses, and what it does not

✅ *"Removing route features raised the model's error by 41–53 %; removing the
congestion proxy changed nothing."*

❌ *"Weather causes 8 % of energy consumption."* Ablation measures what a **model**
loses, not what the **world** does. The groups are mutually confounded — temperature
co-occurs with HVAC load, season and winter driving, and the traffic proxy is derived
from the same speed trace as the route group. **Association only.**
