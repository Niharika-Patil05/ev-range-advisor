# Evidence — E1 - model comparison on real data
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** association-only; accuracy on unseen routes for ONE vehicle model in ONE city over ONE year
**Measured (M):**
- HV battery current and voltage
- vehicle speed
- onboard outside-air temperature
- HVAC power
- SoC
**Derived (D):**
- wh_per_km = -integral(V*I dt) / speed-integrated distance
- route_id by DBSCAN on trip endpoints (500 m)
- congestion proxy = posted limit - observed speed
- aggressiveness = mean positive acceleration
- phys_wh_km from road load over the measured trace
**External (X):**
- eVED map-matched elevation, gradient, speed limits
**Simulated (S):** none
**Assumptions (A):**
- mass = kerb 1493 kg + 80 kg occupant; payload UNOBSERVED
- Cd*A and crr at datasheet values (E7: crr not identifiable)
- regen_eff = 0.55 constant
- rated_range baseline uses the EPA 121 km figure

**Split protocol:** L1 route-wise, 5 folds x 5 seeds, groups disjoint (asserted)
**n:** 480 segments, 134 routes, 3 vehicles (only 2 contribute meaningfully)
**Known confounds:**
- payload unobserved
- temperature co-occurs with HVAC use
- derived routes inherit clustering error
- 2 usable vehicles, so this is not a vehicle-level claim
