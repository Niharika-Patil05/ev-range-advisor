# Evidence — E8 - degradation projection calibration
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** lab 18650 cells; a vehicle pack is not one of these
**Measured (M):**
- discharge capacity per cycle
- ambient temperature per cell
**Derived (D):**
- SoH = capacity / robust reference capacity
- Theil-Sen fade slope per cell
**External (X):** none
**Simulated (S):** none
**Assumptions (A):**
- fade is approximately linear in cycle count over the observed range
- lab cycling represents vehicle use (it does not; see limitations)

**Split protocol:** none - this is a parameter fit; uncertainty from cell-to-cell spread
**n:** 26 cells, 2359 cycles
**Known confounds:**
- ageing protocol is confounded with ambient temperature group
- cell-to-cell spread is 80 % of the mean fade rate
- no DoD or charge-limit variable exists in this dataset
