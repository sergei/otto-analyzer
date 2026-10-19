"""Extract selected PGNs from all YDVR files into per-PGN parquet tables.

Usage: python -m otto.extract <ydvr_root> <out_dir>

Each output table has columns: t (UTC, datetime64[ns]), src, payload (bytes).
Absolute time for each file is anchored on PGN 126992 (System Time).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from otto.ydvr import read_file

PGNS = [
    126992,  # system time (anchor)
    127237,  # heading/track control (autopilot)
    65341,   # Simnet autopilot mode
    65305,   # B&G proprietary status
    130845,  # Simnet parameter handle (raw frames)
    130860,  # Simnet proprietary (raw frames)
    127245,  # rudder
    127250,  # heading
    127251,  # rate of turn
    127252,  # heave
    127257,  # attitude
    128259,  # speed through water
    129025,  # position rapid
    129026,  # COG/SOG rapid
    130306,  # wind
]


def file_offset(d):
    """Seconds to add to file-relative time to get UTC epoch seconds, or None."""
    if 126992 not in d:
        return None
    t, src, pl = d[126992]
    days = pl[:, 2].astype(np.int64) | (pl[:, 3].astype(np.int64) << 8)
    tod = (pl[:, 4].astype(np.int64) | (pl[:, 5].astype(np.int64) << 8)
           | (pl[:, 6].astype(np.int64) << 16) | (pl[:, 7].astype(np.int64) << 24))
    ok = (days < 0xFFFE) & (tod < 0xFFFFFFFE) & (days > 19000)
    if not ok.any():
        return None
    epoch = days[ok] * 86400.0 + tod[ok] * 1e-4
    return float(np.median(epoch - t[ok]))


def process(path):
    d = read_file(path, PGNS)
    off = file_offset(d)
    if off is None:
        return path, None
    out = {}
    for pgn, (t, src, pl) in d.items():
        out[pgn] = pd.DataFrame({
            't': pd.to_datetime(((t + off) * 1e9).astype(np.int64)),
            'src': src,
            'payload': [row.tobytes() for row in pl],
        })
    return path, out


def main(root, out_dir):
    files = sorted(Path(root).glob('YDVR*/*.DAT')) or sorted(Path(root).glob('*.DAT'))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {p: [] for p in PGNS}
    skipped = []
    with Pool() as pool:
        for k, (path, out) in enumerate(pool.imap_unordered(process, files, chunksize=4)):
            if out is None:
                skipped.append(path)
            else:
                for pgn, df in out.items():
                    parts[pgn].append(df)
            if k % 200 == 0:
                print(f'{k}/{len(files)}', flush=True)
    for pgn, dfs in parts.items():
        if dfs:
            df = pd.concat(dfs).sort_values('t', kind='stable').drop_duplicates(['t', 'src', 'payload'])
            df.to_parquet(out_dir / f'pgn_{pgn}.parquet', index=False)
            print(pgn, len(df))
    print('files without time anchor:', len(skipped), [str(p) for p in skipped[:10]])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
