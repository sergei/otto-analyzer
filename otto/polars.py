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

    def bsp(self, tws, twa):
        """Target boat speed (kt); TWS clipped to the table, TWA folded to 0..180."""
        tws = np.clip(np.asarray(tws, float), 0, self._tws_max)
        twa = np.abs(((np.asarray(twa, float) + 180) % 360) - 180)
        return self._f(np.stack([tws, twa], axis=-1))
