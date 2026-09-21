# Evidence — E4 - cross-vehicle generalization (PHEV electric mode)
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** association-only; PHEVs in charge-depleting mode, not BEVs
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- engine RPM
- onboard air temperature
- HVAC power where instrumented
- SoC
**Derived (D):**
- wh_per_km
- electric-mode filter from engine RPM
- route_id by clustering
- physics baseline with per-vehicle mass
**External (X):**
- eVED elevation, gradient, speed limits
**Simulated (S):** none
**Assumptions (A):**
- PHEV mass from VED's BINNED Generalized_Weight (+/-5%)
- Leaf Cd*A, crr, eta applied to all PHEVs -- not re-identified
- missing HVAC channel treated as zero draw
- engine-off guard band 30 s

**Split protocol:** L3c leave-one-vehicle-out, vehicle AND route disjoint, >= 30 segments per vehicle, 5 seeds
**n:** 1764 PHEV-electric segments, 12 vehicles
**Known confounds:**
- HVAC instrumentation differs between fleets (BEV 100%, PHEV A/C 72% heater 12%)
- vehicle parameters are the Leaf's, not re-identified per PHEV model
- engine-off segments may be systematically shorter or gentler than engine-on ones
