# E8 — Calibrating the degradation projection, and deleting what cannot be calibrated

**Data:** 2,359 cycles from 26 real NASA PCoE cells. Provenance `REAL`.
**Trigger:** `project_soh` shipped with six invented constants, and the app turned one
of them into user-facing advice — *"limiting daily charge to 80 % could preserve
roughly X percentage points of SoH."* That was the only place in the project where a
placeholder produced a specific, actionable claim.

## What the cells support

**Base fade rate: 0.00222 SoH per cycle** (Theil-Sen slope per cell, all 26 positive).

| statistic | value |
|---|---|
| mean | 0.00222 |
| 10th / 50th / 90th percentile | 0.00057 / 0.00209 / 0.00346 |
| **cell-to-cell spread** | **80 % of the mean** |

Theil-Sen rather than least squares: several cells open with a partial discharge, and
an OLS slope on those comes out **negative** — a cell that heals with use.

> **The shipped placeholder was 0.00025 per cycle — nine times smaller than measured.**
> The old projection understated degradation by an order of magnitude.

The spread is why the projection is now a **band**, not a line. A single curve would
imply a precision the data does not have.

## What the cells do NOT support — all three removed

| factor | verdict |
|---|---|
| **Arrhenius temperature factor** | **Not identifiable.** `fade ~ ambient` gives r = −0.34, **p = 0.087**, and the sign is the *wrong way round* for an Arrhenius law. The 44 °C group shows the **lowest** fade of all (0.00022 against 0.00315 at 4 °C) — a protocol artefact, not physics. |
| **Depth-of-discharge exponent** | NASA PCoE records **no per-cell DoD variable**. |
| **Charge-cap factor** | NASA PCoE records **no per-cell charge-limit variable**. |

All three are deleted from the code rather than left in place looking plausible.
**A projection that cannot distinguish a good charging habit from a bad one should not
pretend to.**

## The finding that changed the design

Wiring the *measured* rate into the old calendar projection produced nonsense, and the
test suite caught it: at five charges a week, two years is 521 cycles, and
521 × 0.00222 > 1. **The projection predicted a dead pack in two years** — contradicted
by every real EV on the road.

The rate is not wrong. **The extrapolation was.** These cells are cycled at full depth
of discharge and high rate, reaching 80 % SoH in **≈ 90 cycles**; a vehicle pack doing
shallow partial cycles typically takes on the order of **1,000**. The equivalence
factor between a lab cycle and a vehicle cycle is of order ten, and **nothing in this
project measures it.**

> So `project_soh` now reports fade against **lab cycles** and **refuses to convert to
> calendar time.** The app's x-axis is cycles, with the gap stated on screen.

This is a better outcome than the original: the old version gave a confident two-year
calendar forecast built on a rate nine times too small, and its plausibility came
entirely from the error.

## Changes made

- `coupling.FADE_PER_CYCLE` = 0.00222, with P10/P90 for the band.
- `project_soh(soh0, n_cycles, step)` — cycles, not months; band, not line; temperature,
  DoD and charge-cap arguments **removed**.
- `cycles_to_threshold()` added: lab cycles to 80 % SoH, per band.
- **`advisor.advise()` no longer emits charging-habit advice.**
- Battery tab shows the band against lab cycles, with the lab-versus-vehicle gap and
  the removal of charging advice stated on screen.
- Six tests pin all of it, including that the calendar arguments are gone and that lab
  cells reach end of life far faster than a vehicle pack.

## Limitations

- **Lab cells are not vehicle packs.** 2 Ah 18650s in a rig; a pack adds imbalance,
  BMS behaviour and thermal gradients, and is driven rather than cycled.
- **Ageing protocol is confounded with temperature group**, which is part of why the
  temperature factor is not identifiable.
- Fade is treated as **linear in cycle count**, which holds over the observed range but
  ignores the knee that real cells show near end of life.
