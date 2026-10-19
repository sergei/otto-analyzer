"""Expedition polar file: each row 'TWS  TWA1 BSP1  TWA2 BSP2 ...'."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Polar:
    def __init__(self, path):
        rows = []
        for line in open(path):
            if not line.strip() or line.startswith('!'):
                continue
            v = [float(x) for x in line.split()]
            rows.append((v[0], np.array(v[1::2]), np.array(v[2::2])))
        self.tws = np.array([r[0] for r in rows])
        # Resample every TWS row onto a common TWA grid; 0 kt at TWA 0, 0 kt at TWS 0.
        self.twa = np.arange(0.0, 181.0, 1.0)
        grid = np.array([np.interp(self.twa, np.r_[0.0, a], np.r_[0.0, b]) for _, a, b in rows])
        tws = np.r_[0.0, self.tws]
        grid = np.vstack([np.zeros_like(self.twa), grid])
        self._f = RegularGridInterpolator((tws, self.twa), grid, bounds_error=False, fill_value=None)
        self._tws_max = tws[-1]
        # Optimal VMG angles and VMG, tabulated on a fine TWS grid.
        self._vmg_tws = np.arange(0.0, self._tws_max + 0.05, 0.1)
        a = np.arange(25.0, 180.5, 0.5)
        b = self.bsp(np.repeat(self._vmg_tws, len(a)), np.tile(a, len(self._vmg_tws))).reshape(len(self._vmg_tws), -1)
        along = b * np.cos(np.deg2rad(a))
        up = np.where(a < 90, along, -np.inf)
        down = np.where(a > 90, -along, -np.inf)
        self._up_twa, self._up_vmg = a[up.argmax(1)], up.max(1)
        self._down_twa, self._down_vmg = a[down.argmax(1)], down.max(1)

    def bsp(self, tws, twa):
        """Target boat speed (kt); TWS clipped to the table, TWA folded to 0..180."""
        tws = np.clip(np.asarray(tws, float), 0, self._tws_max)
        twa = np.abs(((np.asarray(twa, float) + 180) % 360) - 180)
        return self._f(np.stack([tws, twa], axis=-1))

    def _lookup(self, table, tws):
        tws = np.clip(np.asarray(tws, float), 0, self._tws_max)
        return np.interp(tws, self._vmg_tws, table)

    def upwind_target(self, tws):
        """(optimal upwind TWA deg, target VMG kt) for each TWS."""
        return self._lookup(self._up_twa, tws), self._lookup(self._up_vmg, tws)

    def downwind_target(self, tws):
        """(optimal downwind TWA deg, target VMG kt) for each TWS."""
        return self._lookup(self._down_twa, tws), self._lookup(self._down_vmg, tws)
