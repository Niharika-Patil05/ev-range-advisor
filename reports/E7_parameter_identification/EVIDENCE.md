# Evidence — E7 - road-load parameter identification
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** conditional on model form and fixed assumptions
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- onboard air temperature
- measured HVAC power
**Derived (D):**
- segment energy
- basis integrals
- road grade
**External (X):**
- eVED elevation and gradient
**Simulated (S):** none
**Assumptions (A):**
- mass = kerb + 80.0 kg occupant (payload unobserved)
- Cd*A fixed at 0.661 m^2
- regen_eff fixed at 0.55

**Split protocol:** parameter fit; uncertainty from route-level bootstrap
**n:** 480 segments, 134 routes
**Known confounds:**
- payload unobserved
- crr/eta degeneracy
- aero term not excited at urban speeds
