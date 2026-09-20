# E7 — Road-load parameter identification

**Data:** 480 VED analysis segments, 134 derived routes, 2013 Nissan Leaf. Provenance `REAL`.
**Question 1:** how much of the 22.5 % physics-only error do fitted parameters absorb?
**Question 2:** which parameters can this data identify at all?

---

## Result 1 — fitting the constants buys essentially nothing

| variant | crr | η | MAPE | bias |
|---|---|---|---|---|
| nominal (datasheet) | 0.0090 | 0.890 | **22.54 %** | −4.31 % |
| η only, crr fixed | 0.0090 | 0.884 | 22.53 % | −3.54 % |
| crr + η, `abs_rel` | 0.0053 | 0.793 | **22.38 %** | −3.36 % |
| crr + η, `sq_rel` | 0.0066 | 0.875 | 23.01 % | −10.02 % |
| crr + η, `log_sq` | 0.0044 | 0.735 | 22.83 % | +2.45 % |

**Best fitted MAPE 22.38 % against nominal 22.54 % — an improvement of 0.16 percentage
points.** Two of the three joint fits make MAPE *worse* than the datasheet values.

This is a **negative result, and a useful one.** The 22.5 % error is not caused by
wrong constants. It is caused by physics the model does not contain.

## Result 2 — η is identifiable, `crr` is not

Route-level bootstrap, 200 resamples of whole routes (resampling rows would treat
repeats of one route as independent evidence and give intervals that are far too narrow):

| parameter | median | 95 % CI | CI width |
|---|---|---|---|
| `crr` | 0.0040 | [0.0040, 0.0115] | **187 % of median** |
| `drivetrain_eff` | 0.782 | [0.710, 0.957] | 31.5 % of median |

`corr(crr, η)` across bootstrap samples = **+0.867** — the degeneracy direction.

**Why.** η scales *every* tractive term, including the inertia term that dominates
short stop-go urban segments (median attribution: inertia 160 Wh, roll 144, aero 94,
aux 81, grade 0.3). `crr` scales only the rolling term. So the loss surface is sharp
in η and shallow in `crr`.

A controlled recovery study on real-shaped traces with **known** parameters confirms
it (a legitimate use of simulation: the truth is a value we chose, which no real
dataset can supply — it tests the *estimator*, not the physics). Across independent
noise draws at σ = 0.05, 0.10 and 0.15, η is recovered to within 10 % every time,
while the `crr` estimate ranges over 60–90 % of its true value and often runs to its
bound. The instability is **not monotonic in noise** — at σ = 0.08 it happens to look
stable — which is precisely why a single fit must never be taken as evidence that
`crr` was identified.

> **Recommendation:** hold `crr` at its datasheet value and fit η alone. Report η with
> its interval. **Do not quote a fitted `crr` from this data.**

## Result 3 — the residual is structured, and that is the case for the hybrid

Correlation of `log(measured / physics)` with conditions:

| condition | r |
|---|---|
| `temp_c` | **−0.395** |
| `soc_start` | +0.337 |
| `mean_speed_kmh` | −0.302 |
| `stops_per_km` | +0.286 |
| `aux_power_w` | +0.275 |
| `distance_km` | −0.026 |

The physics model under-predicts more in the cold, at low speed, and with frequent
stops. **No constant `crr` and η can absorb structure of this shape.** This is the
empirical justification for the hybrid's learned residual layer — previously an
architectural assumption, now an observation.

Plausible physical content: temperature-dependent battery internal resistance,
drivetrain efficiency varying with torque and speed rather than being constant,
auxiliary loads beyond the logged HVAC channels, and regenerative braking that
depends on deceleration rate and SoC rather than the fixed `regen_eff = 0.55`.

⚠️ These are **associations on observational data**. They are mutually confounded —
cold weather co-occurs with HVAC use, low speeds and winter driving — and E3 is the
experiment that attempts to separate them. The `soc_start` correlation in particular
should not be over-interpreted: high starting SoC also marks the first trip of a day,
when the cabin is cold and HVAC demand is highest.

## Assumptions this result is conditional on

- **Mass** = kerb 1493 kg + 80 kg nominal occupant. **Payload is unobserved in VED**
  (limitation L5), so mass carries an unquantified error of roughly ±5 %, which maps
  almost directly onto `crr`.
- **Cd·A fixed** at the datasheet 0.661 m². Ann Arbor speeds never excite the aero
  term enough to identify it.
- **`regen_eff` fixed** at 0.55.
- The tractive/regenerative mask is held fixed during fitting (exact given the mask).

## What E7 changes in the project

1. `config.py` keeps its datasheet `crr`. Fitting it is **not** supported by this data.
2. η may be reported as ≈ 0.78–0.88 with a wide interval, and the physics baseline is
   left essentially as-is.
3. The headline physics-only baseline for E1 stays **≈ 22.5 % MAPE**.
4. The hybrid now has an **empirical** motivation, not merely an architectural one.
