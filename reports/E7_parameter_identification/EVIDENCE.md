# Evidence — E7 - road-load parameter identification
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** association-only; parameter estimates are conditional on the model form and the fixed assumptions above
**Measured (M):**
- HV battery current and voltage (segment energy)
- vehicle speed
- onboard outside-air temperature
- HVAC power
**Derived (D):**
- segment energy = -integral(V*I dt)
- road-load basis integrals
- road grade from eVED elevation
**External (X):**
- eVED map-matched elevation and gradient
**Simulated (S):** none
**Assumptions (A):**
- mass = kerb 1493.0 kg + 80.0 kg occupant; payload is UNOBSERVED in VED and maps almost directly onto crr
- Cd*A fixed at datasheet 0.661 m^2; urban speeds do not excite the aero term
- regen efficiency fixed at 0.55
- tractive/regenerative mask held fixed during fitting

**Split protocol:** parameter fit, not a predictive evaluation; uncertainty from a route-level bootstrap (whole routes resampled, never rows)
**n:** 480 segments, 134 routes, 3 vehicles
**Known confounds:**
- payload unobserved (mass error ~+/-5%)
- crr and eta degenerate: corr +0.867 across bootstrap
- temperature co-occurs with HVAC use, low speed and winter driving
