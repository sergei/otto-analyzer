"""Score autopilot settings tests: baseline vs. test blocks against the report's targets.

Usage: python -m otto.blocks <data10hz.parquet> <polars.txt> <blocks.csv> <out_dir> [--pilotvars <parquet>]

blocks.csv has one row per block, times in UTC:
    start,end,label
    2026-11-02 18:00,2026-11-02 18:20,baseline
    2026-11-02 18:20,2026-11-02 18:40,test
The label 'baseline' is the reference; every other label is compared against it.

Per block: autopilot share, conditions, wave-band (7-20 s) swing of actual and commanded
heading, TWA error vs target, excursions > 20 deg, same-setting takeovers, point of sail and
perf_eff (VMG vs target when close-hauled/running, speed vs polar when reaching; otto.sailpoints),
and (with --pilotvars) the pilot's active Perf level and whether gust/heel features reported.
Per label: time-weighted KPIs, the change vs. baseline, and pass/fail against TARGETS.
Blocks with less than MIN_AP_PCT autopilot time are listed but left out of the summary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from otto.kpi import HZ, count_events, prepare
from otto.polars import Polar
from otto.switches import classify_helm_segments, segments
from otto.waveband import band_power, unwrap_deg

EXC_DEG = 20
# Targets from the report's test plan. (kpi, comparison, threshold)
TARGETS = [
    ('hdg_band_rms', '<=', 2.5),
    ('hts_band_rms', '<=', 1.0),
    ('twa_err_rms', '<=', 3.5),
    ('twa_err_p95', '<=', 7.0),
    ('exc_per_24h', '<=', 3.0),
    ('takeovers_per_24h', '<=', 10.0),
]
COND_TWS_KT, COND_TWA_DEG = 2.0, 10.0
# A block that was mostly hand-steered says nothing about the pilot setting under test.
MIN_AP_PCT = 80


def band_rms(x):
    x = pd.Series(x).interpolate(limit=20, limit_direction='both').to_numpy()
    if np.isnan(x).any() or len(x) < 2048:
        return np.nan
    return float(np.sqrt(band_power(unwrap_deg(x))))


def score_block(g, takeover_times, pv):
    ap = g['mode'].isin(['wind', 'compass'])
    ap_hours = ap.sum() / HZ / 3600
    wind = g[g['mode'] == 'wind']
    err = np.where(g['mode'] == 'wind', g.twa_err.abs(), np.where(g['mode'] == 'compass', g.hdg_err.abs(), np.nan))
    n_exc = count_events(ap.to_numpy() & (err > EXC_DEG), min_gap=30 * HZ)
    n_tk = int(((takeover_times >= g.t.iloc[0]) & (takeover_times <= g.t.iloc[-1])).sum())
    r = dict(
        minutes=len(g) / HZ / 60, ap_pct=100 * ap.mean(),
        main_mode=g['mode'].value_counts().index[0],
        tws=g.tws.mean(), twa_abs=g.twa_abs.mean(),
        sail_point=g.sail_point.value_counts().index[0] if g.sail_point.notna().any() else None,
        hdg_band_rms=band_rms(g.hdg.to_numpy()),
        hts_band_rms=band_rms(g.hts.where(ap).to_numpy()) if ap.mean() > 0.9 else np.nan,
        twa_err_rms=float(np.sqrt(np.nanmean(wind.twa_err ** 2))) if wind.twa_err.notna().any() else np.nan,
        twa_err_p95=float(np.nanpercentile(wind.twa_err.abs(), 95)) if wind.twa_err.notna().any() else np.nan,
        excursions=n_exc, takeovers=n_tk, ap_hours=ap_hours,
        perf_eff=float(g.perf_eff.mean()),
        polar_eff=100 * float(np.nanmean(g.stw / g.polar_bsp)),
    )
    if pv is not None:
        p = pv[(pv.t >= g.t.iloc[0]) & (pv.t <= g.t.iloc[-1])]
        r.update(active_perf=p.active_perf.median() if len(p) else np.nan,
                 gust_reported_pct=100 * p.gust_ba.notna().mean() if len(p) else np.nan,
                 heel_comp_reported_pct=100 * p.heel_comp.notna().mean() if len(p) else np.nan)
    return r


def summarize(b):
    def agg(g):
        w = g.minutes
        out = {c: np.average(g[c].dropna(), weights=w[g[c].notna()]) if g[c].notna().any() else np.nan
               for c in ('tws', 'twa_abs', 'ap_pct', 'hdg_band_rms', 'hts_band_rms', 'twa_err_rms', 'twa_err_p95',
                         'perf_eff', 'polar_eff')}
        hours = g.ap_hours.sum()
        out.update(blocks=len(g), minutes=g.minutes.sum(),
                   exc_per_24h=24 * g.excursions.sum() / hours if hours else np.nan,
                   takeovers_per_24h=24 * g.takeovers.sum() / hours if hours else np.nan)
        return pd.Series(out)
    return b.groupby('label').apply(agg, include_groups=False)


def verdicts(s):
    rows = []
    base = s.loc['baseline'] if 'baseline' in s.index else None
    for label, r in s.iterrows():
        for kpi, op, thr in TARGETS:
            v = r[kpi]
            ok = np.nan if np.isnan(v) else (v <= thr if op == '<=' else v >= thr)
            rows.append(dict(label=label, kpi=kpi, value=v, target=f'{op} {thr:g}',
                             vs_baseline=v - base[kpi] if base is not None else np.nan,
                             passes=ok))
        if base is not None and label != 'baseline':
            d = r.perf_eff - base.perf_eff
            rows.append(dict(label=label, kpi='perf_eff', value=r.perf_eff, target='>= baseline',
                             vs_baseline=d, passes=d >= 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('polars')
    ap.add_argument('blocks')
    ap.add_argument('out_dir')
    ap.add_argument('--pilotvars')
    a = ap.parse_args()

    df = prepare(pd.read_parquet(a.data), Polar(a.polars)).reset_index(drop=True)
    seg = segments(df['mode'].to_numpy())
    helm = classify_helm_segments(df, seg) if (seg.kind == 'helm').any() else pd.DataFrame(columns=['t', 'reason'])
    takeover_times = pd.to_datetime(helm.t[helm.reason == 're-engaged, same setting'])
    pv = pd.read_parquet(a.pilotvars) if a.pilotvars else None

    blocks = pd.read_csv(a.blocks, parse_dates=['start', 'end'])
    rows = []
    for _, blk in blocks.iterrows():
        g = df[(df.t >= blk.start) & (df.t < blk.end)]
        if len(g) < 5 * 60 * HZ:
            print(f'skipping {blk.label} block {blk.start}: less than 5 min of data')
            continue
        rows.append(dict(label=blk.label, start=blk.start, end=blk.end, **score_block(g, takeover_times, pv)))
    b = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b.to_csv(out / 'blocks_scored.csv', index=False)
    print(b.drop(columns=['end']).round(2).to_string(index=False))

    low = b.ap_pct < MIN_AP_PCT
    if low.any():
        print(f'\nExcluded from the summary (autopilot < {MIN_AP_PCT:g}% of the block): '
              + ', '.join(f'{r.label} {r.start:%H:%M} ({r.ap_pct:.0f}%)' for r in b[low].itertuples()))
    s = summarize(b[~low])
    s.to_csv(out / 'blocks_summary.csv')
    print('\nPer label (time-weighted; event rates per 24 h of autopilot time):')
    print(s.T.round(2).to_string())

    if 'baseline' in s.index:
        base = s.loc['baseline']
        base_points = set(b[~low & (b.label == 'baseline')].sail_point.dropna())
        for label, r in s.drop(index='baseline').iterrows():
            points = set(b[~low & (b.label == label)].sail_point.dropna())
            if points != base_points:
                print(f'\nWARNING: {label} points of sail {sorted(points)} differ from baseline {sorted(base_points)}; '
                      'perf_eff mixes speed and VMG measures.')
            if abs(r.tws - base.tws) > COND_TWS_KT or abs(r.twa_abs - base.twa_abs) > COND_TWA_DEG:
                print(f'\nWARNING: {label} conditions differ from baseline (TWS {r.tws:.1f} vs {base.tws:.1f} kt, '
                      f'TWA {r.twa_abs:.0f} vs {base.twa_abs:.0f} deg); comparison is not like for like.')
    v = verdicts(s)
    v.to_csv(out / 'blocks_verdict.csv', index=False)
    print('\nTargets:')
    print(v.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
