"""Point of sail relative to the destination, and the performance measure that goes with it.

Point of sail (project skill `sail-points`) is decided by where the DESTINATION lies relative
to the wind, not by the boat's own TWA:
  close-hauled  destination is above the polar's optimal upwind angle (mark TWA < upwind target)
  running       destination is below the polar's optimal downwind angle (mark TWA > downwind target)
  reaching      everything else
On a reach the boat should maximize boat speed; close-hauled and running it must maximize VMG.
VMG is boat speed projected on the wind direction (STW * cos TWA); the destination only picks
the point of sail, it does not enter VMG.

perf_eff (%) is therefore:
  reaching      STW / polar boat speed at the actual TWS, TWA
  close-hauled  upwind VMG (STW * cos TWA) / polar target upwind VMG at TWS
  running       downwind VMG (-STW * cos TWA) / polar target downwind VMG at TWS
"""
import numpy as np

# PacCup 2026 finish off Kaneohe Bay, Oahu (where the track ends its run in, 2026-07-19 ~05:20 UTC).
FINISH = (21.48, -157.78)


def bearing_deg(lat, lon, dest=FINISH):
    """Initial great-circle bearing (deg true) from each position to dest."""
    p1, l1 = np.deg2rad(lat), np.deg2rad(lon)
    p2, l2 = np.deg2rad(dest[0]), np.deg2rad(dest[1])
    dl = l2 - l1
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.rad2deg(np.arctan2(y, x)) % 360


def add_sail_points(df, polar, dest=FINISH):
    """Add mark_brg, mark_twa, sail_point, vmg, target angles and perf_eff to a 10 Hz frame.

    Needs columns lat, lon, hdg (true), twa (signed), tws, stw and polar_bsp.
    """
    df = df.copy()
    twd = (df.hdg + df.twa) % 360
    df['mark_brg'] = bearing_deg(df.lat.to_numpy(), df.lon.to_numpy(), dest)
    df['mark_twa'] = np.abs((twd - df.mark_brg + 180) % 360 - 180)
    up_twa, up_vmg = polar.upwind_target(df.tws.to_numpy())
    down_twa, down_vmg = polar.downwind_target(df.tws.to_numpy())
    df['target_up_twa'], df['target_down_twa'] = up_twa, down_twa
    df['sail_point'] = np.select([df.mark_twa < up_twa, df.mark_twa > down_twa], ['close-hauled', 'running'],
                                 'reaching')
    df.loc[df.mark_twa.isna() | df.tws.isna(), 'sail_point'] = None
    along = df.stw * np.cos(np.deg2rad(df.twa.abs()))
    df['vmg'] = np.where(df.sail_point == 'close-hauled', along,
                         np.where(df.sail_point == 'running', -along, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        df['perf_eff'] = 100 * np.select(
            [df.sail_point == 'close-hauled', df.sail_point == 'running', df.sail_point == 'reaching'],
            [along / up_vmg, -along / down_vmg, df.stw / df.polar_bsp], np.nan)
    df['perf_eff'] = df.perf_eff.replace([np.inf, -np.inf], np.nan)
    return df
