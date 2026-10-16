"""Large autopilot excursions, broach-like events and rudder balance.

Usage: python -m otto.excursions <race10hz.parquet> <polars.txt> <out_dir>

1. Excursions: autopilot stretches where the error exceeds EXC_DEG (TWA error vs target in
   wind mode, heading error vs commanded heading in compass mode), away from tacks/gybes.
   Each event gets its UTC time (for video review), peak error, direction (round-up when TWA
   was below its 2-minute median) and whether the crew took the helm within 60 s.
2. Broach-like events: |roll| > ROLL_DEG together with a round-up > ROUNDUP_DEG, counted the
   same way for every mode, overall and downwind (TWA > 130) in 14-22 kt.
3. Rudder balance downwind (TWA > 120): mean bear-away rudder by mode and TWS, and mean
   rudder per tack (a difference between tacks hints at a rudder zero offset).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from otto.kpi import HZ, count_events, prepare
from otto.polars import Polar

EXC_DEG = 20
MERGE_S = 30
ROLL_DEG, ROUNDUP_DEG = 35, 20
TWS_BINS = [0, 14, 18, 22, 40]


def excursions(df):
    near_maneuver = df.maneuver.rolling(60 * HZ, center=True, min_periods=1).max().astype(bool).to_numpy()
    err = np.where(df['mode'] == 'wind', df.twa_err.abs(), df.hdg_err.abs())
    ap = df['mode'].isin(['wind', 'compass']).to_numpy()
    idx = np.flatnonzero(ap & (err > EXC_DEG) & ~near_maneuver)
    events = []
    for i in idx:
        if events and i - events[-1][1] <= MERGE_S * HZ:
            events[-1][1] = i
        else:
            events.append([i, i])
    mode = df['mode'].to_numpy()
    rows = []
    for a, b in events:
        k = a + int(np.nanargmax(err[a:b + 1]))
        around = df.iloc[max(a - 10 * HZ, 0):b + 10 * HZ]
        rows.append(dict(
            t_utc=df.t.iloc[k], mode=mode[k], peak_err=err[k], twa=df.twa.iloc[k], target_twa=df.target_twa.iloc[k],
            direction='round-up' if df.twa_dev.iloc[k] < 0 else 'bear-away',
            max_roll=np.nanmax(np.abs(around.roll)), max_rudder=np.nanmax(np.abs(around.rudder)),
            tws=around.tws.mean(), dur_s=(b - a) / HZ,
            helm_within_60s=bool((mode[b:b + 60 * HZ] == 'helm').any())))
    return pd.DataFrame(rows).sort_values('peak_err', ascending=False)


def broach_rates(df):
    near_maneuver = df.maneuver.rolling(120 * HZ, center=True, min_periods=1).max().astype(bool)
    ev = ((df.roll.abs() > ROLL_DEG) & (df.twa_dev < -ROUNDUP_DEG) & ~near_maneuver).to_numpy()
    downwind = (df.twa_abs.rolling(120 * HZ, center=True, min_periods=1).mean() > 130).to_numpy() \
        & df.tws.between(14, 22).to_numpy()
    rows = []
    for m in ('wind', 'compass', 'helm'):
        sel = (df['mode'] == m).to_numpy()
        for scope, mask in (('all', sel), ('downwind 14-22 kt', sel & downwind)):
            n, hours = count_events(ev & mask, min_gap=60 * HZ), mask.sum() / HZ / 3600
            rows.append(dict(mode=m, scope=scope, events=n, hours=hours, per_24h=24 * n / hours if hours else np.nan))
    return pd.DataFrame(rows)


def rudder_balance(df):
    d = df[df['mode'].isin(['wind', 'compass', 'helm']) & (df.twa_abs > 120)].copy()
    # Starboard tack (TWA > 0): bearing away = turning to port = negative rudder.
    d['bearaway_rudder'] = np.where(d.twa > 0, -d.rudder, d.rudder)
    d['tws_bin'] = pd.cut(d.tws, TWS_BINS)
    by_tws = d.groupby(['tws_bin', 'mode'], observed=True).bearaway_rudder.mean().unstack('mode')
    by_tack = d.groupby(np.where(d.twa > 0, 'starboard', 'port')).rudder.mean()
    return by_tws, by_tack


def main(data, polar_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_parquet(data), Polar(polar_path)).reset_index(drop=True)

    e = excursions(df)
    e.to_csv(out / 'ap_excursions.csv', index=False)
    hours = df['mode'].value_counts() / HZ / 3600
    counts = e.groupby(['mode', 'direction']).size().unstack(fill_value=0)
    counts['per_24h'] = 24 * counts.sum(axis=1) / hours.reindex(counts.index)
    print(f'{len(e)} autopilot excursions > {EXC_DEG} deg; {int(e.helm_within_60s.sum())} followed by a takeover '
          f'within 60 s; {int((e.peak_err > 40).sum())} over 40 deg')
    print(counts.round(2).to_string())
    print(e.head(15).round(1).to_string(index=False))

    b = broach_rates(df)
    b.to_csv(out / 'broach_rates.csv', index=False)
    print(f'\nbroach-like events (|roll| > {ROLL_DEG} and round-up > {ROUNDUP_DEG} deg):')
    print(b.round(2).to_string(index=False))

    by_tws, by_tack = rudder_balance(df)
    by_tws.to_csv(out / 'bearaway_rudder.csv')
    print('\nmean bear-away rudder downwind (deg), by TWS:')
    print(by_tws.round(2).to_string())
    print('\nmean rudder by tack downwind (deg, + = starboard):')
    print(by_tack.round(2).to_string())
    print(f'implied rudder zero offset: {by_tack.mean():.2f} deg')


if __name__ == '__main__':
    main(*sys.argv[1:4])
