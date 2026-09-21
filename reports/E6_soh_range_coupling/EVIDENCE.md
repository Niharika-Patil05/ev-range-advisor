# Evidence — E6 - SoH to range propagation (RQ3)
**Provenance:** `COUPLED-SIM`  **Conclusion type:** `SIMULATED`  **Claim strength:** propagation study; NOT real-world end-to-end validation
**Measured (M):**
- VED pack current and voltage (consumption error)
- NASA charge curves and discharge capacity (SoH error)
**Derived (D):**
- out-of-fold relative errors from E1 and E5
- log-space variance decomposition
**External (X):** none
**Simulated (S):**
- the COUPLING between a lab 18650 cell and a vehicle pack
- 20000 Monte-Carlo draws from the measured SoH error distribution
- temperature-derate coefficient swept over 0.003-0.009 per degC
**Assumptions (A):**
- error sources independent (they are not perfectly so)
- the lab-cell SoH error distribution applies to a vehicle pack
- nominal pack energy 24 kWh, unchanged by ageing

**Split protocol:** errors are out-of-fold: route-wise for consumption, leave-one-battery-out for SoH
**n:** 491 consumption errors, 2359 SoH errors
**Known confounds:**
- lab cells are not vehicle packs (no imbalance, BMS, gradients)
- NASA cells are 2 Ah 18650s, the Leaf pack is 24 kWh
- SoH error has a heavy tail: worst cell 13.9 pp in E5

> ⚠️ **COUPLED-SIM.** Real inputs propagated through a simulated coupling between datasets that cannot be joined row-wise. This is a propagation study, **not** real-world end-to-end validation.
