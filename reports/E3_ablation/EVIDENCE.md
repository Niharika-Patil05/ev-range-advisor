# Evidence — E3 - feature-group ablation
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** ASSOCIATION ONLY. Dropping a feature group changes the MODEL's error; it does not measure a causal effect on consumption. Groups are mutually confounded.
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- onboard outside-air temperature
- HVAC power
- SoC
**Derived (D):**
- wh_per_km
- stops, speed statistics, aggressiveness
- congestion proxy = posted limit - observed speed
- elevation gain/loss
- physics baseline
**External (X):**
- eVED elevation, gradient, speed limits (OSM)
**Simulated (S):** none
**Assumptions (A):**
- mass = kerb + 80 kg occupant
- route tolerance 500 m

**Split protocol:** L1 route-wise, 5 folds x 5 seeds
**n:** 480 segments, 134 routes, 2 usable vehicles
**Known confounds:**
- temperature co-occurs with HVAC load, season and winter driving
- the traffic proxy is derived from speed, so it overlaps the route group
- the hybrid's physics layer still sees dropped variables internally
