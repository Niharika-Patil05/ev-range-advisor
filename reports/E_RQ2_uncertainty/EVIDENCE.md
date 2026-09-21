# Evidence — RQ2 - conditional coverage of the range band
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** coverage measured on held-out routes for one vehicle model
**Measured (M):**
- HV battery current and voltage
- onboard air temperature
- vehicle speed
- HVAC power
**Derived (D):**
- wh_per_km
- hybrid point prediction
- temperature tertiles
**External (X):**
- eVED elevation, gradient, speed limits
**Simulated (S):** none
**Assumptions (A):**
- alpha = 0.10
- tertile edges fixed in advance (HYPOTHESES.md)
- conformal assumes exchangeability, which shift violates

**Split protocol:** route-disjoint train/calibration/test 50/25/25, 5 seeds
**n:** 480 segments, 134 routes
**Known confounds:**
- cold segments co-occur with high HVAC load
- calibration sets are small (~120 segments), so group quantiles rest on few points
