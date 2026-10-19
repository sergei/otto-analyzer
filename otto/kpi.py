"""Per-window steering KPIs and condition-matched autopilot vs. helm comparison.

Usage: python -m otto.kpi <race10hz.parquet> <polars.txt> <out_dir>

Windows are 10 minutes. A window is kept only if >= 98 % of it is in one mode
(wind / compass / helm), >= 80 % of it on one point of sail, and it contains no tack or gybe.

KPIs computed identically for every mode ("comparable"):
  twa_dev_rms   RMS of |TWA| about its 2-minute centred median (deg)
  hdg_dev_rms   RMS of heading about its 2-minute centred median (deg)
  exc15_ph      excursions per hour with |TWA deviation| > 15 deg
  exc25_ph      same, > 25 deg
  rot_rms       RMS yaw rate (deg/s)
  roll_p99      99th percentile of |roll - median roll| (deg)
  rud_rate_rms  RMS rudder rate (deg/s)
  rud_rev_pm    rudder reversals per minute (direction changes after >= 1 deg of travel)
  rud_abs_mean  mean |rudder| (deg)
  rud_sat_pct   % of time |rudder| >= 20 deg
  polar_eff     mean STW / polar boat speed at the actual TWA (%)
  perf_eff      performance for the point of sail (otto.sailpoints): VMG / target VMG when
                close-hauled or running, STW / polar speed when reaching (%)
Autopilot only:
  twa_err_*     TWA error vs. autopilot target TWA (wind mode)
  hdg_err_*     heading error vs. heading-to-steer (both autopilot modes)
  drive_lag_s   lag of rudder position behind rudder order
  drive_err_rms RMS(rudder position - order shifted by drive lag)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from otto.polars import Polar
from otto.sailpoints import add_sail_points

HZ = 10
WINDOW = '10min'
MEDIAN_S = 120


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def count_events(mask, min_gap=5 * HZ):
    """Number of separate True runs, merging runs closer than min_gap samples."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    return int(1 + np.sum(np.diff(idx) > min_gap))


def reversals(x, hysteresis=1.0):
    """Direction changes of x after at least `hysteresis` travel (zig-zag count)."""
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 0
    n, direction, extreme = 0, 0, x[0]
    for v in x[1:]:
        if direction == 0:
            if abs(v - extreme) >= hysteresis:
                direction, extreme = (1 if v > extreme else -1), v
        elif direction > 0:
            if v > extreme:
                extreme = v
            elif v < extreme - hysteresis:
                n, direction, extreme = n + 1, -1, v
        else:
            if v < extreme:
                extreme = v
            elif v > extreme + hysteresis:
                n, direction, extreme = n + 1, 1, v
    return n


def best_lag(order, position, max_lag=3 * HZ):
    ok = ~(np.isnan(order) | np.isnan(position))
    if ok.sum() < 60 * HZ or np.nanstd(order) < 0.5:
        return np.nan, np.nan
    o, p = order - np.nanmean(order), position - np.nanmean(position)
    o[~ok] = 0
    p[~ok] = 0
    best = None
    for lag in range(0, max_lag + 1):
        c = np.dot(o[:len(o) - lag], p[lag:])
        if best is None or c > best[0]:
            best = (c, lag)
    lag = best[1]
    err = position[lag:] - order[:len(order) - lag]
    return lag / HZ, float(np.sqrt(np.nanmean(err ** 2)))


def prepare(df, polar):
    df = df.copy()
    df['twa_abs'] = df.twa.abs()
    df['target_abs'] = df.target_twa.abs()
    df['polar_bsp'] = polar.bsp(df.tws.to_numpy(), df.twa_abs.to_numpy())
    w = MEDIAN_S * HZ
    med = df.twa_abs.rolling(w, center=True, min_periods=w // 2).median()
    df['twa_dev'] = df.twa_abs - med
    # Heading median via unit vectors to survive the 0/360 wrap.
    rad = np.deg2rad(df.hdg)
    s = np.sin(rad).rolling(w, center=True, min_periods=w // 2).mean()
    c = np.cos(rad).rolling(w, center=True, min_periods=w // 2).mean()
    df['hdg_dev'] = wrap180(df.hdg - np.rad2deg(np.arctan2(s, c)))
    rud = df.rudder.rolling(3, center=True, min_periods=1).mean()
    df['rud_rate'] = rud.diff() * HZ
    # Positive TWA error = sailing deeper (further off the wind) than the target.
    df['twa_err'] = np.where(np.sign(df.twa) == np.sign(df.target_twa), df.twa_abs - df.target_abs, np.nan)
    df['hdg_err'] = wrap180(df.hdg - df.hts)
    # Maneuver = sign change of 10 s smoothed TWA.
    side = np.sign(df.twa.rolling(10 * HZ, center=True, min_periods=1).mean())
    df['maneuver'] = side.ne(side.shift()) & side.shift().notna()
    return add_sail_points(df, polar)


def window_kpis(g):
    r = {'t': g.t.iloc[0], 'n': len(g)}
    counts = g['mode'].value_counts()
    r['mode'] = counts.index[0]
    r['mode_frac'] = counts.iloc[0] / len(g)
    r['maneuver'] = bool(g.maneuver.any())
    points = g.sail_point.value_counts()
    r['sail_point'] = points.index[0] if len(points) else None
    r['sail_point_frac'] = points.iloc[0] / len(g) if len(points) else 0.0
    hours = len(g) / HZ / 3600
    r.update(
        tws=g.tws.mean(), twa_abs=g.twa_abs.mean(), stw=g.stw.mean(),
        pitch_std=g.pitch.std(), roll_std=g.roll.std(),
        lat=g.lat.mean(), lon=g.lon.mean(),
        twa_dev_rms=np.sqrt(np.nanmean(g.twa_dev ** 2)),
        hdg_dev_rms=np.sqrt(np.nanmean(g.hdg_dev ** 2)),
        exc15_ph=count_events(g.twa_dev.abs().to_numpy() > 15) / hours,
        exc25_ph=count_events(g.twa_dev.abs().to_numpy() > 25) / hours,
        rot_rms=np.sqrt(np.nanmean(g.rot ** 2)),
        roll_p99=np.nanpercentile((g.roll - g.roll.median()).abs(), 99),
        rud_rate_rms=np.sqrt(np.nanmean(g.rud_rate ** 2)),
        rud_rev_pm=reversals(g.rudder.to_numpy()) / (len(g) / HZ / 60),
        rud_abs_mean=g.rudder.abs().mean(),
        rud_sat_pct=100 * (g.rudder.abs() >= 20).mean(),
        polar_eff=100 * (g.stw / g.polar_bsp).replace([np.inf, -np.inf], np.nan).mean(),
        perf_eff=g.perf_eff.mean(), mark_twa=g.mark_twa.mean(),
    )
    if r['mode'] in ('wind', 'compass'):
        e = g.hdg_err
        r.update(hdg_err_mean=e.mean(), hdg_err_rms=np.sqrt(np.nanmean(e ** 2)),
                 hdg_err_p95=np.nanpercentile(e.abs(), 95) if e.notna().any() else np.nan,
                 off_hdg_pct=100 * g.off_heading_exceeded.mean())
        r['drive_lag_s'], r['drive_err_rms'] = best_lag(g.rudder_order.to_numpy(), g.rudder.to_numpy())
    if r['mode'] == 'wind':
        e = g.twa_err
        r.update(target_twa=g.target_abs.mean(), twa_err_mean=e.mean(), twa_err_rms=np.sqrt(np.nanmean(e ** 2)),
                 twa_err_p95=np.nanpercentile(e.abs(), 95) if e.notna().any() else np.nan,
                 twa_err_in5_pct=100 * (e.abs() <= 5).mean(),
                 exc15_target_ph=count_events(e.abs().to_numpy() > 15) / hours,
                 exc25_target_ph=count_events(e.abs().to_numpy() > 25) / hours)
    return r


TWS_BINS = [0, 12, 15, 18, 21, 40]
TWA_BINS = [0, 70, 110, 135, 155, 181]
COMPARABLE = ['perf_eff', 'polar_eff', 'twa_dev_rms', 'hdg_dev_rms', 'exc15_ph', 'exc25_ph', 'rot_rms', 'roll_p99',
              'rud_rate_rms', 'rud_rev_pm', 'rud_abs_mean', 'rud_sat_pct']
# Rare-event KPIs are aggregated with the mean; the median of a mostly-zero rate is uninformative.
EVENT_KPIS = ['exc15_ph', 'exc25_ph', 'rud_sat_pct']
HIGHER_IS_BETTER = {'perf_eff', 'polar_eff'}
# Windows are matched on wind speed, point of sail (by destination) and sea state. Not on the
# boat's own TWA: on a run, the angle steered is part of the VMG performance being compared.
MATCH_KEYS = ['tws_bin', 'sail_point', 'sea_bin']


def add_bins(w):
    w = w.copy()
    w['tws_bin'] = pd.cut(w.tws, TWS_BINS, right=False)
    w['twa_bin'] = pd.cut(w.twa_abs, TWA_BINS, right=False)
    w['sea_bin'] = pd.qcut(w.pitch_std, 3, labels=['calm', 'moderate', 'rough'])
    return w


def matched_comparison(w, ap_mode, min_n=3):
    """Condition-matched difference ap_mode - helm, weighted by the autopilot's time in each bin."""
    keys = MATCH_KEYS
    grouped = w.groupby(keys + ['mode'], observed=True)
    med = grouped[COMPARABLE].median()
    med[EVENT_KPIS] = grouped[EVENT_KPIS].mean()
    cnt = w.groupby(keys + ['mode'], observed=True).size()
    rows = []
    for key, n_ap in cnt.xs(ap_mode, level='mode').items():
        try:
            n_h = cnt.loc[key + ('helm',)]
        except KeyError:
            continue
        if n_ap >= min_n and n_h >= min_n:
            rows.append((n_ap, n_h, med.loc[key + (ap_mode,)], med.loc[key + ('helm',)]))
    if not rows:
        return None
    weights = np.array([r[0] for r in rows], float)
    ap = pd.DataFrame([r[2] for r in rows])
    helm = pd.DataFrame([r[3] for r in rows])
    out = pd.DataFrame({
        ap_mode: np.average(ap, axis=0, weights=weights),
        'helm': np.average(helm, axis=0, weights=weights),
    }, index=COMPARABLE)
    out['diff'] = out[ap_mode] - out['helm']
    out['ap_worse_bins_pct'] = [100 * np.average((ap[c].values > helm[c].values) if c not in HIGHER_IS_BETTER
                                                 else (ap[c].values < helm[c].values), weights=weights)
                                for c in COMPARABLE]
    out.attrs['bins'] = len(rows)
    out.attrs['ap_windows'] = int(weights.sum())
    out.attrs['helm_windows'] = int(sum(r[1] for r in rows))
    return out


def main(data, polar_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_parquet(data), Polar(polar_path))
    rows = [window_kpis(g) for _, g in df.groupby(df.t.dt.floor(WINDOW)) if len(g) > 0.9 * 600 * HZ]
    w = pd.DataFrame(rows)
    w.to_csv(out_dir / 'windows_all.csv', index=False)
    clean = w[(w.mode_frac >= 0.98) & ~w.maneuver & w['mode'].isin(['wind', 'compass', 'helm'])
              & (w.sail_point_frac >= 0.8)]
    clean = add_bins(clean)
    clean.to_csv(out_dir / 'windows.csv', index=False)
    print('windows total', len(w), 'clean', len(clean))
    print(clean.groupby('mode').size())
    summary = clean.groupby('mode')[['tws', 'twa_abs', 'pitch_std'] + COMPARABLE].median()
    summary[EVENT_KPIS] = clean.groupby('mode')[EVENT_KPIS].mean()
    print(summary.T.round(2).to_string())
    print('\nperf_eff by point of sail (median of windows, n):')
    print(clean.groupby(['sail_point', 'mode']).perf_eff.agg(['median', 'size']).unstack('mode').round(1).to_string())
    ap_cols = ['target_twa', 'twa_err_mean', 'twa_err_rms', 'twa_err_p95', 'twa_err_in5_pct',
               'hdg_err_mean', 'hdg_err_rms', 'hdg_err_p95', 'drive_lag_s', 'drive_err_rms']
    ap_events = ['exc15_target_ph', 'exc25_target_ph', 'off_hdg_pct']
    ap = clean[clean['mode'] != 'helm'].groupby('mode')
    print(pd.concat([ap[ap_cols].median(), ap[ap_events].mean()], axis=1).T.round(2).to_string())
    for m in ('wind', 'compass'):
        cmp = matched_comparison(clean, m)
        if cmp is not None:
            print(f'\n== {m} vs helm, matched over {cmp.attrs["bins"]} bins '
                  f'({cmp.attrs["ap_windows"]} AP / {cmp.attrs["helm_windows"]} helm windows)')
            print(cmp.round(2).to_string())
            cmp.to_csv(out_dir / f'matched_{m}_vs_helm.csv')


if __name__ == '__main__':
    main(*sys.argv[1:4])
