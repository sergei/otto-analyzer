---
name: sail-points
description: How to split sailing data into points of sail (close-hauled, reaching, running) and which performance measure to use on each - boat speed vs polar when reaching, VMG = BS * cos(TWA) vs polar target VMG when close-hauled or running. Use whenever judging boat speed, autopilot vs helm performance, or polar efficiency.
---

Distinguish the following sail points 
- close-hauled - destination above target upwind angle (~< 50 TWA) 
- running - destination below downwind target angle (~> 130 TWA)                   
- reaching - everything else 

While on a reaching course the boat's speed itself is maximized, but on close-hauled and running courses the VMG must be maximized 
The VMG is the projection of the boat speed on the wind direction, i.e. VMG = BS * cos(TWA)

## How this is applied (otto/sailpoints.py)

- **The destination only picks the point of sail.** Its bearing relative to the true wind direction
  ("mark TWA") is compared with the polar's optimal upwind and downwind angles at the current TWS.
  The boat's own TWA does not decide the point of sail: a boat heading for a dead-downwind mark is
  running even while it sails 145° TWA on a gybe.
- **VMG does not depend on the destination.** It is boat speed projected on the wind direction,
  `VMG = BS * cos(TWA)` (positive upwind; downwind use `-BS * cos(TWA)`). It is not velocity toward
  the mark.
- **Targets come from the polar** at the current TWS: target upwind/downwind VMG is the maximum of
  `polar_BS(TWS, a) * cos(a)` over angles `a`, and the angle where it occurs is the optimal angle.
- **Performance (`perf_eff`, %):**
  - close-hauled: `100 * BS * cos(TWA) / target upwind VMG`
  - running: `100 * -BS * cos(TWA) / target downwind VMG`
  - reaching: `100 * BS / polar_BS(TWS, TWA)`
- Compare autopilot and helm only within the same point of sail; mixing points mixes a speed
  measure with a VMG measure.
