# Evidence — E5 - SoH estimation, leave-one-battery-out
**Provenance:** `REAL`  **Conclusion type:** `DEMONSTRATED`  **Claim strength:** lab 18650 cells under controlled cycling; NOT a vehicle pack
**Measured (M):**
- charge voltage, current and temperature curves
- discharge capacity (LABEL ONLY)
**Derived (D):**
- constant-current and constant-voltage durations
- time in a fixed voltage window
- incremental-capacity peak
- temperature rise
- charge energy accepted
- SoH = capacity / robust reference capacity
**External (X):** none
**Simulated (S):** none
**Assumptions (A):**
- reference capacity = median of the 5 largest plausible capacities, not the first (9 cells begin with a partial cycle)
- capacities outside 0.2-2.2 Ah treated as measurement failures
- health indicators taken from the PRECEDING charge curve

**Split protocol:** leave-one-battery-out; each cell predicted by a model that never saw it
**n:** 2359 cycles, 26 cells
**Known confounds:**
- cells were aged at different ambient temperatures (4-44 C)
- protocols differ across cell groups
- lab cells lack pack effects: imbalance, BMS logic, thermal gradients
