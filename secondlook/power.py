"""Package power (RAPL) and CPU package temperature from sysfs. Returns None when unreadable.

RAPL energy_uj is root-readable on recent kernels. To allow reading without sudo for the demo:
    sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
"""
import glob
import time

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


class Power:
    def __init__(self, idle_watts=None, rapl=RAPL, thermal_glob="/sys/class/thermal/thermal_zone*"):
        self.idle = idle_watts
        self.rapl = rapl
        self._last = None
        self.watts = None
        self._temp_path = None
        for zone in sorted(glob.glob(thermal_glob)):
            if _read(zone + "/type") in ("x86_pkg_temp", "TCPU", "cpu"):
                self._temp_path = zone + "/temp"
                break

    def sample(self):
        """Call regularly; updates self.watts (package watts averaged since the last call)."""
        e, t = _read(self.rapl), time.time()
        if e is None:
            return None
        e = int(e)
        if self._last and t > self._last[1] and e >= self._last[0]:
            self.watts = (e - self._last[0]) / 1e6 / (t - self._last[1])
        self._last = (e, t)
        return self.watts

    def above_idle(self):
        if self.watts is None or self.idle is None:
            return None
        return self.watts - self.idle

    def temp_c(self):
        v = _read(self._temp_path) if self._temp_path else None
        return int(v) / 1000 if v else None
