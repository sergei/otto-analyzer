"""Does the TWA on the bus include a leeway correction?

Usage: python -m otto.leeway <race10hz.parquet> <out_dir>

Recomputes TWA from AWA, AWS and boat speed through the water with no leeway, and compares
it with the TWA the H5000 CPU sends. B&G models leeway as K * heel / BSP^2, so if the
correction is on, the difference follows heel / BSP^2. Writes the difference by point of sail.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_STW = 3
MAX_DELTA = 30
TWA_BANDS = [('upwind', 0, 60), ('reach', 60, 135), ('downwind', 135, 181)]


def twa_no_leeway(awa, aws, stw):
    a = np.radians(awa)
    return np.degrees(np.arctan2(aws * np.sin(a), aws * np.cos(a) - stw))


def leeway_check(df):
    df = df.dropna(subset=['awa', 'aws', 'stw', 'twa', 'roll'])
    delta = (df.twa - twa_no_leeway(df.awa, df.aws, df.stw) + 180) % 360 - 180
    ok = (df.stw > MIN_STW) & (delta.abs() < MAX_DELTA)
    df, delta = df[ok], delta[ok]
    x = df.roll / df.stw ** 2
    rows = []
    for name, lo, hi in [('all', 0, 181)] + TWA_BANDS:
        m = (df.twa.abs() >= lo) & (df.twa.abs() < hi)
        rows.append({'band': name, 'n': int(m.sum()), 'median_delta': delta[m].median(),
                     'median_abs_delta': delta[m].abs().median(), 'r_heel_bsp2': np.corrcoef(x[m], delta[m])[0, 1]})
    k, _ = np.polyfit(x, delta, 1)
    return pd.DataFrame(rows), k


def main(data, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data, columns=['awa', 'aws', 'stw', 'twa', 'roll'])
    t, k = leeway_check(df)
    t.to_csv(out / 'leeway.csv', index=False)
    print('bus TWA minus TWA with no leeway (deg):')
    print(t.round(3).to_string(index=False))
    print(f'fitted K (delta = K * heel / BSP^2): {k:.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
