# Evidence — E2 - distribution-shift ladder (RQ1)
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** association-only; shift robustness for ONE vehicle model in ONE city. L3 rests on n=2 and is indicative only.
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- onboard outside-air temperature
- HVAC power
- SoC
**Derived (D):**
- wh_per_km
- route_id by clustering
- physics baseline
- congestion proxy
- aggressiveness
**External (X):**
- eVED elevation, gradient, speed limits
**Simulated (S):** none
**Assumptions (A):**
- mass = kerb + 80 kg occupant
- route clustering tolerance 500 m
- cold = lowest tertile of segment-mean onboard temperature, fixed before the experiment (HYPOTHESES.md)

**Split protocol:** L0 random / L1 route-wise / L2 temperature regime / L3 leave-one-vehicle-out / L3b vehicle AND route disjoint; 5 seeds
**n:** 480 segments, 134 routes, 2 usable vehicles of 3
**Known confounds:**
- cold segments co-occur with high HVAC load and winter driving
- L2 discards ~42% of data to keep routes disjoint
- vehicle 541 contributes only 10 segments
- L3 leaves 16-30% of test segments on routes seen in training; L3b removes that and is the stricter claim
