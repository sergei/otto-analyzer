"""Build a 10 Hz race timeline from the extracted PGN tables.

Usage: python -m otto.dataset <raw_dir> <out.parquet>

Sources (identified from address claims and content):
  15  B&G autopilot computer: 127237 steering mode / heading-to-steer, 127245 rudder order,
      65341 autopilot mode (byte 4: 2 = heading hold, 3 = true wind angle; bytes 6-7 target TWA)
  19  Navico heading sensor: magnetic heading, rate of turn, attitude, heave
  27  B&G H5000 CPU: calibrated wind (apparent and true), boat speed
  30  Simrad rudder feedback: rudder position, 20 Hz
  127 B&G GPS: position, COG/SOG
  128 second compass, used only for magnetic variation
"""
import sys

import numpy as np
import pandas as pd

from otto import decode as D

RACE_START = pd.Timestamp('2026-07-07 18:05')
RACE_END = pd.Timestamp('2026-07-19 05:25')
TOL = pd.Timedelta('0.5s')


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def attach(grid, df, cols, tol=TOL):
    df = df[(df.t >= RACE_START - tol) & (df.t <= RACE_END + tol)].sort_values('t')
    df = df.assign(t=df.t.astype('datetime64[ns]'))
    return pd.merge_asof(grid, df[['t'] + cols], on='t', direction='nearest', tolerance=tol)


def build(raw):
    grid = pd.DataFrame({'t': pd.date_range(RACE_START, RACE_END, freq='100ms').astype('datetime64[ns]')})

    htc = D.heading_track_control(raw)
    ap = D.simnet_ap_mode(raw)
    ap['target_twa'] = D.u16(np.stack([ap.b6.values, ap.b7.values], 1), 0, 1e-4 * D.RAD)
    grid = attach(grid, htc, ['steering_mode', 'hts', 'off_heading_exceeded', 'rudder_limit_exceeded'], pd.Timedelta('1s'))
    grid = attach(grid, ap.rename(columns={'b4': 'ap_sub'}), ['ap_sub', 'target_twa'], pd.Timedelta('1s'))

    grid = attach(grid, D.rudder(raw, 15).rename(columns={'order': 'rudder_order'}), ['rudder_order'], pd.Timedelta('1s'))
    grid = attach(grid, D.rudder(raw, 30).rename(columns={'position': 'rudder'}), ['rudder'])

    grid = attach(grid, D.heading(raw, 19).rename(columns={'heading': 'hdg_mag'}), ['hdg_mag'])
    var = D.heading(raw, 128)[['t', 'variation']].dropna()
    grid = attach(grid, var, ['variation'], pd.Timedelta('10s'))
    grid['variation'] = grid.variation.interpolate(limit_direction='both')
    grid['hdg'] = (grid.hdg_mag + grid.variation) % 360

    grid = attach(grid, D.rate_of_turn(raw, 19), ['rot'])
    grid = attach(grid, D.attitude(raw, 19), ['pitch', 'roll'])
    grid = attach(grid, D.heave(raw, 19), ['heave'])

    w = D.wind(raw, 27)
    grid = attach(grid, w[w.ref == 2].rename(columns={'ws': 'aws', 'wa': 'awa'}), ['aws', 'awa'])
    grid = attach(grid, w[w.ref == 3].rename(columns={'ws': 'tws', 'wa': 'twa'}), ['tws', 'twa'])
    grid = attach(grid, w[w.ref == 0].rename(columns={'wa': 'twd'}), ['twd'])
    grid = attach(grid, D.speed(raw, 27), ['stw'], pd.Timedelta('1s'))

    grid = attach(grid, D.cog_sog(raw, 127), ['cog', 'sog'], pd.Timedelta('1s'))
    grid = attach(grid, D.position(raw, 127), ['lat', 'lon'], pd.Timedelta('2s'))

    # Signed angles: -180..180, negative = port.
    for c in ('awa', 'twa', 'target_twa'):
        grid[c] = wrap180(grid[c])

    engaged = grid.steering_mode == 4
    grid['mode'] = np.select(
        [engaged & (grid.ap_sub == 3), engaged & (grid.ap_sub == 2), grid.steering_mode == 0],
        ['wind', 'compass', 'helm'], 'unknown')
    for c in grid.columns:
        if grid[c].dtype == np.float64:
            grid[c] = grid[c].astype(np.float32)
    return grid


if __name__ == '__main__':
    df = build(sys.argv[1])
    df.to_parquet(sys.argv[2], index=False)
    print(df.describe().T.to_string())
    print(df['mode'].value_counts() / 36000)  # hours
