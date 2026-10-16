"""Autopilot <-> helm switches.

Usage: python -m otto.switches <race10hz.parquet> <polars.txt> <out_dir>

1. Paired comparison: the last SPAN of an autopilot segment vs. the first SPAN of the
   adjacent helm segment (and vice versa). Wind, sea and sail plan are nearly identical
   across a switch, so the difference isolates who (or what) is steering. SETTLE seconds
   next to the switch are skipped; spans with a tack/gybe or TWS change >= 2 kt are dropped.
2. Takeovers: every autopilot -> helm switch, with what the boat was doing in the
   PRE seconds before it, compared to the same statistics at random autopilot times.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from otto.kpi import HZ, prepare
from otto.polars import Polar

SPAN = 5 * 60 * HZ
SETTLE = 30 * HZ
PRE = 60 * HZ
SHORT_HELM_MIN = 5


def segments(mode):
    change = np.flatnonzero(mode[1:] != mode[:-1]) + 1
    seg = pd.DataFrame({'start': np.r_[0, change], 'end': np.r_[change, len(mode)]})
    seg['mode'] = mode[seg.start]
    # Merge compass/wind runs into one autopilot segment; keep the dominant AP mode.
    seg['kind'] = np.where(seg['mode'] == 'helm', 'helm', np.where(seg['mode'] == 'unknown', 'unknown', 'ap'))
    grp = (seg.kind != seg.kind.shift()).cumsum()
    seg['len'] = seg.end - seg.start
    out = seg.groupby(grp).agg(start=('start', 'first'), end=('end', 'last'), kind=('kind', 'first'))
    dom = seg.groupby([grp, 'mode']).len.sum().reset_index().sort_values('len').groupby('kind').last()
    out['ap_mode'] = dom['mode'].reindex(out.index).where(out.kind == 'ap')
    return out.reset_index(drop=True)


def side_stats(g):
    return dict(
        polar_eff=100 * np.nanmean(g.stw / g.polar_bsp),
        stw=g.stw.mean(), tws=g.tws.mean(), twa_abs=g.twa_abs.mean(),
        twa_dev_rms=np.sqrt(np.nanmean(g.twa_dev ** 2)), rot_rms=np.sqrt(np.nanmean(g.rot ** 2)),
        roll_p99=np.nanpercentile((g.roll - g.roll.median()).abs(), 99),
        rud_rate_rms=np.sqrt(np.nanmean(g.rud_rate ** 2)),
    )


def pre_stats(g):
    return dict(
        max_abs_twa_err=np.nanmax(np.abs(g.twa_err)) if g.twa_err.notna().any() else np.nan,
        max_abs_hdg_err=np.nanmax(np.abs(g.hdg_err)) if g.hdg_err.notna().any() else np.nan,
        max_abs_twa_dev=np.nanmax(np.abs(g.twa_dev)),
        max_abs_roll_dev=np.nanmax(np.abs(g.roll - g.roll.median())),
        max_abs_rot=np.nanmax(np.abs(g.rot)),
        max_abs_rudder=np.nanmax(np.abs(g.rudder)),
        tws=g.tws.mean(), twa_abs=g.twa_abs.mean(),
    )


def paired(df, seg):
    rows = []
    for k in range(len(seg) - 1):
        s1, s2 = seg.iloc[k], seg.iloc[k + 1]
        if {s1.kind, s2.kind} != {'ap', 'helm'}:
            continue
        i = s2.start
        if i - s1.start < SPAN + SETTLE or s2.end - i < SPAN + SETTLE:
            continue
        before, after = df.iloc[i - SETTLE - SPAN:i - SETTLE], df.iloc[i + SETTLE:i + SETTLE + SPAN]
        if before.maneuver.any() or after.maneuver.any():
            continue
        sb, sa = side_stats(before), side_stats(after)
        if abs(sb['tws'] - sa['tws']) >= 2:
            continue
        ap_seg = s1 if s1.kind == 'ap' else s2
        ap_side, helm_side = (sb, sa) if s1.kind == 'ap' else (sa, sb)
        rows.append(dict(t=df.t.iloc[i], switch=f'{s1.kind}->{s2.kind}', ap_mode=ap_seg.ap_mode,
                         tws=(sb['tws'] + sa['tws']) / 2, twa_abs=(sb['twa_abs'] + sa['twa_abs']) / 2,
                         **{f'{c}_ap': v for c, v in ap_side.items()},
                         **{f'{c}_helm': v for c, v in helm_side.items()}))
    s = pd.DataFrame(rows)
    for c in ('polar_eff', 'stw', 'twa_dev_rms', 'rot_rms', 'roll_p99', 'rud_rate_rms'):
        s[f'{c}_diff'] = s[f'{c}_ap'] - s[f'{c}_helm']
    return s


def takeovers(df, seg, rng):
    rows = []
    for k in range(len(seg) - 1):
        s1, s2 = seg.iloc[k], seg.iloc[k + 1]
        if s1.kind != 'ap' or s2.kind != 'helm' or s2.start - s1.start < PRE:
            continue
        g = df.iloc[s2.start - PRE:s2.start]
        rows.append(dict(t=df.t.iloc[s2.start], ap_mode=df['mode'].iloc[s2.start - 1],
                         helm_min=(s2.end - s2.start) / HZ / 60, **pre_stats(g)))
    tk = pd.DataFrame(rows)
    # Baseline: random 60 s autopilot stretches that end at least 2 minutes before any takeover.
    ap_idx = np.flatnonzero(df['mode'].isin(['wind', 'compass']).to_numpy())
    takeover_idx = seg.start[(seg.kind == 'helm') & (seg.kind.shift() == 'ap')].to_numpy()
    base = []
    for i in rng.choice(ap_idx, 3000, replace=False):
        if i < PRE or (np.abs(takeover_idx - i) < 2 * 60 * HZ).any():
            continue
        g = df.iloc[i - PRE:i]
        if not g['mode'].isin(['wind', 'compass']).all():
            continue
        base.append(dict(ap_mode=df['mode'].iloc[i], **pre_stats(g)))
    return tk, pd.DataFrame(base)


def main(data, polar_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_parquet(data), Polar(polar_path)).reset_index(drop=True)
    seg = segments(df['mode'].to_numpy())

    s = paired(df, seg)
    s.to_csv(out_dir / 'switches.csv', index=False)
    print('usable paired switches:', len(s))
    print(s.groupby(['ap_mode', 'switch']).size())
    diff_cols = [c for c in s.columns if c.endswith('_diff')]
    for m, g in s.groupby('ap_mode'):
        d = g.polar_eff_diff
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'\n{m}: n={len(g)}  polar_eff AP-helm mean {d.mean():.2f} pts (95% CI +-{1.96 * se:.2f}), '
              f'median {d.median():.2f}, AP slower in {100 * (d < 0).mean():.0f}% of switches')
        print(g[diff_cols].describe().T[['mean', '50%', 'std']].round(2).to_string())
        print(g.groupby(pd.cut(g.tws, [0, 14, 18, 22, 40]), observed=True).polar_eff_diff
              .agg(['size', 'mean', 'median']).round(2).to_string())

    tk, base = takeovers(df, seg, np.random.default_rng(0))
    tk.to_csv(out_dir / 'takeovers.csv', index=False)
    ap_hours = df['mode'].isin(['wind', 'compass']).sum() / HZ / 3600
    print(f'\ntakeovers: {len(tk)} in {ap_hours:.0f} AP hours = {24 * len(tk) / ap_hours:.1f} per 24 h; '
          f'{(tk.helm_min < SHORT_HELM_MIN).sum()} handed back within {SHORT_HELM_MIN} min')
    print(tk.groupby('ap_mode').size())
    cols = ['max_abs_twa_err', 'max_abs_hdg_err', 'max_abs_twa_dev', 'max_abs_roll_dev', 'max_abs_rot', 'max_abs_rudder']
    short = tk[tk.helm_min < SHORT_HELM_MIN]
    print('\n60 s before takeover (median) vs random AP 60 s (median):')
    print(pd.DataFrame({'short takeovers': short[cols].median(), 'all takeovers': tk[cols].median(),
                        'random AP': base[cols].median(),
                        'random AP p90': base[cols].quantile(0.9)}).round(2).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:4])
