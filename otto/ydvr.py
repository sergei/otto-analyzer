"""Reader for Yacht Devices YDVR-04 .DAT recordings.

Record layout (little endian), as observed in our recordings:
  u16  time      milliseconds within the current minute (0..59999)
  u32  header    29-bit CAN identifier; 0xFFFFFFFF marks a service record
  data:
    service record              8 bytes
    PGN 59904 (ISO request)     3 bytes
    standard fast-packet PGN    seq byte, length byte, then the assembled payload
    everything else             8 bytes (proprietary fast-packet PGNs are raw frames)
"""
import struct

import numpy as np

# Standard (non-proprietary) fast-packet PGNs, which the recorder stores assembled.
FAST_PACKET = {
    126208, 126464, 126983, 126984, 126985, 126986, 126987, 126988, 126996, 126998,
    127233, 127237, 127489, 127496, 127497, 127498, 127503, 127504, 127506, 127507,
    127509, 127510, 127511, 127512, 127513, 127514, 128275, 128520, 129029, 129038,
    129039, 129040, 129041, 129044, 129045, 129284, 129285, 129301, 129302, 129538,
    129540, 129541, 129542, 129545, 129547, 129549, 129551, 129556, 129792, 129793,
    129794, 129795, 129796, 129797, 129798, 129799, 129800, 129801, 129802, 129803,
    129804, 129805, 129806, 129807, 129808, 129809, 129810, 130060, 130061, 130064,
    130065, 130066, 130067, 130068, 130069, 130070, 130071, 130072, 130073, 130074,
    130320, 130321, 130322, 130323, 130324, 130567, 130577, 130578,
}


def pgn_of(can_id):
    pf = (can_id >> 16) & 0xFF
    dp = (can_id >> 24) & 1
    ps = (can_id >> 8) & 0xFF
    return (dp << 16) | (pf << 8) | (ps if pf >= 240 else 0)


def iter_records(buf):
    """Yield (rel_seconds, pgn, src, payload) for every CAN record in a file.

    rel_seconds counts from the first minute of the file; minute rollovers are
    detected from the millisecond counter going backwards.
    """
    i, n = 0, len(buf)
    minute, prev_ms = 0, None
    unpack = struct.unpack_from
    while i + 6 <= n:
        ms, hdr = unpack('<HI', buf, i)
        i += 6
        if prev_ms is not None and ms < prev_ms:
            minute += 1
        prev_ms = ms
        if hdr == 0xFFFFFFFF:
            i += 8
            continue
        can_id = hdr & 0x1FFFFFFF
        pgn = pgn_of(can_id)
        if pgn == 59904:
            length, start = 3, i
        elif pgn in FAST_PACKET:
            if i + 2 > n:
                break
            length, start = buf[i + 1], i + 2
            i += 2
        else:
            length, start = 8, i
        i += length
        if i > n:
            break
        yield minute * 60 + ms / 1000.0, pgn, can_id & 0xFF, buf[start:start + length]


def read_file(path, pgns):
    """Read one .DAT file, keeping only the requested PGNs.

    Returns {pgn: (t_rel float64[N], src uint8[N], payload uint8[N, W])} where W is
    the longest payload for that PGN (shorter payloads are padded with 0xFF).
    """
    with open(path, 'rb') as f:
        buf = f.read()
    rows = {p: [] for p in pgns}
    for t, pgn, src, data in iter_records(buf):
        r = rows.get(pgn)
        if r is not None:
            r.append((t, src, data))
    out = {}
    for pgn, r in rows.items():
        if not r:
            continue
        width = max(len(d) for _, _, d in r)
        payload = np.full((len(r), width), 0xFF, dtype=np.uint8)
        for k, (_, _, d) in enumerate(r):
            payload[k, :len(d)] = np.frombuffer(d, dtype=np.uint8)
        out[pgn] = (np.array([x[0] for x in r]), np.array([x[1] for x in r], dtype=np.uint8), payload)
    return out
