"""Vectorised field decoders for the PGNs we use (layouts per NMEA 2000 / canboat)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAD = 180.0 / np.pi


def load(raw_dir, pgn, src=None):
    df = pd.read_parquet(Path(raw_dir) / f'pgn_{pgn}.parquet')
    if src is not None:
        df = df[df.src == src]
    width = max(len(p) for p in df.payload) if len(df) else 8
    pl = np.full((len(df), width), 0xFF, dtype=np.uint8)
    for k, p in enumerate(df.payload):
        pl[k, :len(p)] = np.frombuffer(p, dtype=np.uint8)
    return df.t.to_numpy(), df.src.to_numpy(), pl


def u16(pl, i, scale=1.0):
    v = pl[:, i].astype(np.int64) | (pl[:, i + 1].astype(np.int64) << 8)
    return np.where(v >= 0xFFFD, np.nan, v * scale)


def i16(pl, i, scale=1.0):
    v = (pl[:, i].astype(np.int64) | (pl[:, i + 1].astype(np.int64) << 8))
    v = np.where(v >= 0x8000, v - 0x10000, v)
    return np.where(v >= 0x7FFD, np.nan, v * scale)


def i32(pl, i, scale=1.0):
    v = (pl[:, i].astype(np.int64) | (pl[:, i + 1].astype(np.int64) << 8)
         | (pl[:, i + 2].astype(np.int64) << 16) | (pl[:, i + 3].astype(np.int64) << 24))
    v = np.where(v >= 0x80000000, v - 0x100000000, v)
    return np.where(v >= 0x7FFFFFFD, np.nan, v * scale)


def frame(t, src, **cols):
    return pd.DataFrame({'t': t, 'src': src, **cols})


def heading_track_control(raw):
    """127237 from the autopilot."""
    t, s, p = load(raw, 127237)
    return frame(
        t, s,
        rudder_limit_exceeded=p[:, 0] & 3,
        off_heading_exceeded=(p[:, 0] >> 2) & 3,
        override=(p[:, 0] >> 6) & 3,
        steering_mode=p[:, 1] & 7,
        turn_mode=(p[:, 1] >> 3) & 7,
        heading_ref=(p[:, 1] >> 6) & 3,
        rudder_dir_order=(p[:, 2] >> 5) & 7,
        rudder_order=i16(p, 3, 1e-4 * RAD),
        hts=u16(p, 5, 1e-4 * RAD),
        track=u16(p, 7, 1e-4 * RAD),
        rudder_limit=u16(p, 9, 1e-4 * RAD),
        off_heading_limit=u16(p, 11, 1e-4 * RAD),
        rot_order=i16(p, 15, 3.125e-5 * RAD),
        vessel_heading=u16(p, 19, 1e-4 * RAD),
    )


def rudder(raw, src=None):
    """127245: angle order and position, degrees (+ = starboard)."""
    t, s, p = load(raw, 127245, src)
    return frame(t, s, dir_order=p[:, 1] & 7, order=i16(p, 2, 1e-4 * RAD), position=i16(p, 4, 1e-4 * RAD))


def heading(raw, src=None):
    t, s, p = load(raw, 127250, src)
    return frame(t, s, heading=u16(p, 1, 1e-4 * RAD), deviation=i16(p, 3, 1e-4 * RAD),
                 variation=i16(p, 5, 1e-4 * RAD), ref=p[:, 7] & 3)


def rate_of_turn(raw, src=None):
    t, s, p = load(raw, 127251, src)
    return frame(t, s, rot=i32(p, 1, 3.125e-8 * RAD))  # deg/s


def attitude(raw, src=None):
    t, s, p = load(raw, 127257, src)
    return frame(t, s, yaw=i16(p, 1, 1e-4 * RAD), pitch=i16(p, 3, 1e-4 * RAD), roll=i16(p, 5, 1e-4 * RAD))


def heave(raw, src=None):
    t, s, p = load(raw, 127252, src)
    return frame(t, s, heave=i16(p, 1, 0.01))


def speed(raw, src=None):
    t, s, p = load(raw, 128259, src)
    return frame(t, s, stw=u16(p, 1, 0.01 * 1.943844))  # knots


def cog_sog(raw, src=None):
    t, s, p = load(raw, 129026, src)
    return frame(t, s, cog_ref=p[:, 1] & 3, cog=u16(p, 2, 1e-4 * RAD), sog=u16(p, 4, 0.01 * 1.943844))


def position(raw, src=None):
    t, s, p = load(raw, 129025, src)
    return frame(t, s, lat=i32(p, 0, 1e-7), lon=i32(p, 4, 1e-7))


def wind(raw, src=None):
    """130306: speed knots, angle degrees, ref 0=true N, 1=magnetic, 2=apparent, 3=true boat, 4=true water."""
    t, s, p = load(raw, 130306, src)
    return frame(t, s, ws=u16(p, 1, 0.01 * 1.943844), wa=u16(p, 3, 1e-4 * RAD), ref=p[:, 5] & 7)


def simnet_ap_mode(raw):
    """65341: raw bytes kept; meaning established empirically."""
    t, s, p = load(raw, 65341)
    return frame(t, s, **{f'b{k}': p[:, k] for k in range(2, p.shape[1])})
