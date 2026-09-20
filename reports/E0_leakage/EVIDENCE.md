# Evidence — E0 - leakage demonstration
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** direct comparison of evaluation protocols on identical data
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- onboard air temperature
- HVAC power
- SoC
**Derived (D):**
- wh_per_km
- route_id by DBSCAN on trip endpoints
- congestion proxy
- physics baseline
**External (X):**
- eVED elevation, gradient and speed limits
**Simulated (S):** none
**Assumptions (A):**
- vehicle mass = kerb + 80 kg occupant
- route clustering tolerance 500 m

**Split protocol:** L0 random rows vs L1 unseen routes, 5 folds, 5 seeds
**n:** 480 segments, 134 routes
**Known confounds:**
- derived routes inherit clustering error
- only 2 vehicles contribute meaningfully
