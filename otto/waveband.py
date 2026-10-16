"""Wave-band steering analysis: does the pilot make the boat snake?

Usage: python -m otto.waveband <race10hz.parquet> <windows.csv> <out_dir>
           [--twa-min 135] [--tws-min 14] [--tws-max 20]

For every clean 10-minute window (from otto.kpi) in the selected conditions:
  *_band_rms     RMS amplitude in the 7-20 s wave band (0.05-0.15 Hz), deg
                 hts = autopilot commanded heading, hdg = actual heading, twa, awa
  rot_band_rms   same for yaw rate, deg/s
  coh_twa_roll   mean coherence of TWA with roll in the band (mast-motion contamination)
  coh_hts_twa    mean coherence of commanded heading with TWA (pilot following wind noise)
  rud_rot_corr0  correlation of rudder with yaw rate at zero lag (> 0: rudder drives the yaw)
Summary per mode: median of the band powers (as RMS) and of the coherences.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from otto.kpi import HZ

BAND = (0.05, 0.15)
NPERSEG = 1024


def unwrap_deg(a):
    return np.rad2deg(np.unwrap(np.deg2rad(a)))


def band_power(x):
    f, p = signal.welch(signal.detrend(x), fs=HZ, nperseg=NPERSEG)
    sel = (f >= BAND[0]) & (f < BAND[1])
    return np.trapezoid(p[sel], f[sel])


def band_coherence(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):  # constant input (e.g. a held heading) gives 0/0
        f, c = signal.coherence(signal.detrend(x), signal.detrend(y), fs=HZ, nperseg=NPERSEG)
    return c[(f >= BAND[0]) & (f < BAND[1])].mean()


def window_stats(x, mode):
    sig = {'hdg': unwrap_deg(x.hdg), 'twa': x.twa.abs().to_numpy(), 'awa': x.awa.abs().to_numpy(),
           'rot': x.rot.to_numpy()}
    if mode != 'helm' and not x.hts.isna().any():
        sig['hts'] = unwrap_deg(x.hts)
    r = {f'{k}_band_pow': band_power(v) for k, v in sig.items()}
    roll = x.roll.to_numpy()
    r['coh_twa_roll'] = band_coherence(sig['twa'], roll)
    if 'hts' in sig:
        r['coh_hts_twa'] = band_coherence(sig['hts'], sig['twa'])
    r['rud_rot_corr0'] = np.corrcoef(signal.detrend(x.rudder.to_numpy()), signal.detrend(sig['rot']))[0, 1]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('windows')
    ap.add_argument('out_dir')
    ap.add_argument('--twa-min', type=float, default=135)
    ap.add_argument('--tws-min', type=float, default=14)
    ap.add_argument('--tws-max', type=float, default=20)
    a = ap.parse_args()

    df = pd.read_parquet(a.data, columns=['t', 'hdg', 'hts', 'twa', 'awa', 'roll', 'rot', 'rudder']).set_index('t')
    w = pd.read_csv(a.windows, parse_dates=['t'])
    w = w[(w.twa_abs > a.twa_min) & w.tws.between(a.tws_min, a.tws_max)]
    rows = []
    for _, win in w.iterrows():
        x = df.loc[win.t:win.t + pd.Timedelta('10min')].interpolate(limit=20)
        if x[['hdg', 'twa', 'awa', 'roll', 'rot', 'rudder']].isna().any().any():
            continue
        rows.append(dict(t=win.t, mode=win['mode'], tws=win.tws, twa_abs=win.twa_abs, **window_stats(x, win['mode'])))
    r = pd.DataFrame(rows)
    for c in [c for c in r.columns if c.endswith('_band_pow')]:
        r[c.replace('_pow', '_rms')] = np.sqrt(r.pop(c))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / 'waveband.csv', index=False)

    rms = [c for c in r.columns if c.endswith('_band_rms')]
    # Median of band power reported as RMS, so the summary is not biased by averaging square roots.
    summary = r.groupby('mode')[rms].agg(lambda s: np.sqrt(np.nanmedian(s ** 2)) if s.notna().any() else np.nan)
    summary = summary.join(r.groupby('mode')[['coh_twa_roll', 'coh_hts_twa', 'rud_rot_corr0']].median())
    summary['n'] = r.groupby('mode').size()
    print(f'Wave band {1 / BAND[1]:.0f}-{1 / BAND[0]:.0f} s, TWA > {a.twa_min:g}, '
          f'TWS {a.tws_min:g}-{a.tws_max:g} kt:')
    print(summary.T.round(2).to_string())


if __name__ == '__main__':
    main()
