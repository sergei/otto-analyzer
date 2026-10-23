---
name: score-autopilot-test
description: Score an on-water H5000 autopilot settings test (baseline vs. test blocks) from a new YDVR-04 recording against the KPI targets in the Sun Dragon autopilot report. Use when the user has sailed a settings test, has a new YDVR log to analyze, or asks whether a pilot setting change helped.
---

# Score an autopilot settings test

The test plan alternates a baseline setting and a test setting in ~20-minute blocks in the
same conditions. This skill turns the YDVR recording plus the block times into a verdict.

## 1. Collect inputs

Ask for anything not already given:

- **YDVR folder** with the test recording (folders `YDVR*/` of `.DAT` files, or a flat folder of `.DAT` files).
- **Block times**, start/end in **UTC**, each labelled `baseline` or a short test name (e.g. `damping6`).
  If the user gives local times, convert to UTC and say which offset you used.
- **Which setting changed** in the test blocks, and its baseline value (for the write-up only).
  The race baseline (controller photos plus the boat's settings sheet, raw export in
  `data/H5000_settings_sheet.csv`) is in `data/H5000_autopilot_settings-2026-10-01.csv`;
  if the user has a newer settings file, use that.

Pick a short test id from the date and setting, e.g. `2026-11-02-damping`, and use
`out/tests/<id>/` for every output.

## 2. Write the blocks file

`out/tests/<id>/blocks.csv`:

```csv
start,end,label
2026-11-02 18:00,2026-11-02 18:20,baseline
2026-11-02 18:20,2026-11-02 18:40,damping6
```

## 3. Run the pipeline

From the repo root, with `<window_start>`/`<window_end>` spanning all blocks (a few minutes
of margin is fine):

```sh
T=out/tests/<id>
.venv/bin/python -m otto.extract   <ydvr_folder> $T/raw
.venv/bin/python -m otto.pilotvars <ydvr_folder> $T/pilotvars.parquet
.venv/bin/python -m otto.dataset   $T/raw $T/data10hz.parquet '<window_start>' '<window_end>'
.venv/bin/python -W ignore -m otto.blocks $T/data10hz.parquet data/sun_dragon_expedition_polars_v3.txt \
    $T/blocks.csv $T --pilotvars $T/pilotvars.parquet
```

`otto.extract` takes a few seconds per hour of recording (the whole 12-day race took ~4 minutes). If it reports files without a time anchor, the
GPS was not sending PGN 126992 — stop and tell the user.

## 4. Check before judging

- **Settings actually changed**: in `blocks_scored.csv`, `active_perf`, `gust_reported_pct` and
  `heel_comp_reported_pct` should differ between baseline and test blocks when the test was a
  Perf, gust-response or heel-compensation change. If they don't, say so — the setting may not
  have been applied. (Damping and motion-correction changes don't show in pilot variables.)
- **Excluded blocks**: blocks with < 80% autopilot time are left out; list them.
- **Conditions**: if the script prints a conditions or point-of-sail WARNING, the comparison is not like
  for like (`perf_eff` measures VMG when running but boat speed when reaching).
- **Noise floor**: on race data with no setting change, baseline vs. "test" blocks differed by
  about 0.1° in heading swing and 0.1 polar points. Treat differences below ~0.3° swing or ~1 `perf_eff`
  point (with 4 blocks each) as no change. Event rates (excursions, takeovers) from a few hours
  are very noisy; report counts, not just rates.

## 5. Report

Give the user a short table per test label: the KPI, baseline value, test value, change, target,
pass/fail, from `blocks_verdict.csv`. Then one or two sentences: did the setting move the KPI it
was meant to move (see the report's Recommendations table), and should they keep it?

Targets (from the report's test plan, also in `otto/blocks.py` `TARGETS`):

| KPI | Race value (wind mode) | Target |
| --- | --- | --- |
| `hdg_band_rms` heading swing, 7-20 s band | 4.5° | ≤ 2.5° |
| `hts_band_rms` commanded-heading swing, 7-20 s band | 3.75° | ≤ 1° |
| `twa_err_rms` / `twa_err_p95` | 3.2° / 6.1° | ≤ 3.5° / ≤ 7° |
| `exc_per_24h` excursions > 20° | 6.5 | ≤ 3 |
| `takeovers_per_24h` same-setting takeovers | 33 | ≤ 10 |
| `perf_eff` (VMG running/close-hauled, speed reaching) | — | ≥ baseline |

Do not edit the shared report doc or commit anything unless the user asks.
