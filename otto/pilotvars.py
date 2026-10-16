"""Extract H5000 pilot internal variables from B&G key-value PGN 130824 (source 15).

Usage: python -m otto.pilotvars <ydvr_root> <out.parquet>

130824 is a proprietary fast-packet PGN, stored by the YDVR as raw frames. After the
2-byte manufacturer code the payload is a list of items: u16 (key | length << 12), value.
Keys sent by the H5000 Pilot CPU, in the order the 2018 release notes list the variables:
  0x17c active_perf      Active Perf mode chosen by the pilot (1..5)
  0x17d gust_ba          Gust bear-away (deg)
  0x17e tws_ba           TWS bear-away (deg)
  0x17f heel_comp        Heel compensation rudder (deg)
  0x180 net_course       Pilot net course (deg, magnetic)
  0x181 pilot_target_wa  Pilot target wind angle (deg)
  0x182 weather_helm     Pilot weather helm (deg rudder)
  0x183 mean_heel        Pilot mean heel (deg)
"""
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from otto.extract import file_offset
from otto.ydvr import iter_records, read_file

PILOT_SRC = 15
KEYS = {0x17c: 'active_perf', 0x17d: 'gust_ba', 0x17e: 'tws_ba', 0x17f: 'heel_comp', 0x180: 'net_course',
        0x181: 'pilot_target_wa', 0x182: 'weather_helm', 0x183: 'mean_heel'}
RAD = 180.0 / np.pi


def parse_items(payload):
    out, i = {}, 2
    while i + 2 <= len(payload):
        word = struct.unpack_from('<H', payload, i)[0]
        key, n = word & 0xFFF, word >> 12
        i += 2
        if n == 0 or i + n > len(payload):
            break
        out[key] = payload[i:i + n]
        i += n
    return out


def value(key, raw):
    if key == 0x17c:
        return np.nan if raw[0] == 0xFF else float(raw[0])
    v = struct.unpack('<h', raw)[0]
    return np.nan if v >= 0x7FFD else v * 1e-4 * RAD


def process(path):
    buf = open(path, 'rb').read()
    off = file_offset(read_file(path, [126992]))
    if off is None:
        return []
    asm, rows = {}, []
    for t, pgn, src, d in iter_records(buf):
        if pgn != 130824 or src != PILOT_SRC:
            continue
        fc, seq = d[0] & 0x1F, d[0] >> 5
        if fc == 0:
            asm[seq] = [d[1], bytearray(d[2:8]), 0]
        elif seq in asm and asm[seq][2] + 1 == fc:
            asm[seq][1] += d[1:8]
            asm[seq][2] = fc
        else:
            continue
        a = asm.get(seq)
        if a and len(a[1]) >= a[0]:
            items = parse_items(bytes(a[1][:a[0]]))
            del asm[seq]
            row = {'t': t + off}
            row.update({name: value(k, items[k]) for k, name in KEYS.items() if k in items})
            rows.append(row)
    return rows


def main(root, out):
    files = sorted(Path(root).glob('YDVR*/*.DAT'))
    rows = []
    with Pool() as pool:
        for r in pool.imap_unordered(process, files, chunksize=4):
            rows.extend(r)
    df = pd.DataFrame(rows).sort_values('t')
    df['t'] = pd.to_datetime((df.t * 1e9).astype(np.int64))
    df.to_parquet(out, index=False)
    print(df.describe().T.to_string())


if __name__ == '__main__':
    main(*sys.argv[1:3])
