# otto-analyzer

Quantifies B&G H5000 autopilot performance on Sun Dragon (Jeanneau SunFast 3300) during
PacCup 2026, from the NMEA 2000 log recorded by a Yacht Devices YDVR-04.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
YDVR=/Users/sergei/SailingVideos/2026-PACCUP-2026/20-DATA/10-YDVR
POLAR=data/sun_dragon_expedition_polars_v3.txt

.venv/bin/python -m otto.extract   $YDVR out/raw                  # ~4 min: per-PGN parquet tables
.venv/bin/python -m otto.pilotvars $YDVR out/pilotvars.parquet    # H5000 pilot internal variables
.venv/bin/python -m otto.dataset   out/raw out/race10hz.parquet   # 10 Hz race timeline
.venv/bin/python -m otto.kpi       out/race10hz.parquet $POLAR out/kpi
.venv/bin/python -m otto.switches  out/race10hz.parquet $POLAR out/kpi
```

Everything is written under `out/`, which is not committed.

| Output | Contents |
| --- | --- |
| `out/kpi/windows.csv` | KPIs per 10-minute window (single mode, no tack/gybe) with condition bins |
| `out/kpi/matched_*_vs_helm.csv` | Autopilot vs helm, matched by TWS x TWA x sea-state bin |
| `out/kpi/switches.csv` | Paired comparison 5 min either side of each autopilot/helm switch |
| `out/kpi/takeovers.csv` | Each autopilot -> helm switch and the boat's state in the 60 s before |

## Modules

| Module | Purpose |
| --- | --- |
| `otto/ydvr.py` | YDVR-04 `.DAT` reader |
| `otto/extract.py` | Extracts selected PGNs to parquet, anchored to UTC via PGN 126992 |
| `otto/decode.py` | Vectorised field decoders for the PGNs used |
| `otto/dataset.py` | Merges sources onto a 10 Hz grid; infers autopilot mode; sets the race window |
| `otto/polars.py` | Expedition polar lookup (target boat speed by TWS, TWA) |
| `otto/kpi.py` | Window KPIs and condition-matched comparison |
| `otto/switches.py` | Paired switch comparison and takeover analysis |
| `otto/pilotvars.py` | H5000 pilot variables (active Perf, gust/TWS bear-away, heel comp, ...) from PGN 130824 |

## YDVR-04 record format

As observed in these recordings (little endian):

- `u16` milliseconds within the minute, `u32` CAN identifier (`0xFFFFFFFF` = service record, 8 data bytes).
- Standard fast-packet PGNs are stored assembled: sequence byte, length byte, payload.
- Proprietary fast-packet PGNs (130816+) are stored as raw 8-byte frames.
- PGN 59904 records carry 3 bytes; everything else 8.

## Sources on the bus

| Source | Device | Used for |
| --- | --- | --- |
| 15 | B&G H5000 Pilot CPU | 127237 steering mode / heading-to-steer, 127245 rudder order, 65341 mode + target TWA, 130824 pilot variables |
| 19 | Navico heading sensor | Magnetic heading, rate of turn, attitude |
| 27 | B&G H5000 CPU | Calibrated apparent/true wind, boat speed |
| 30 | Simrad rudder feedback | Rudder position, 20 Hz |
| 127 | B&G GPS | Position, COG/SOG, system time |
| 128 | Second compass | Magnetic variation only |

Autopilot mode: 127237 steering mode 4 = engaged, 0 = standby (helm). When engaged,
65341 byte 4 = 2 is heading hold (compass) and 3 is true-wind-angle mode, with the
target TWA in bytes 6-7. These meanings, and the 130824 key assignments, are inferred
from the data and the H5000 2018 release notes, not from B&G documentation of the PGNs.

## Not yet scripted

The wave-band spectral analysis, takeover classification and excursion list in the
report were run ad hoc and are not in this repo yet.
