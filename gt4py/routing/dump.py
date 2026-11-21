"""Reader for the routing dumps of WaSiM's WASIM_ROUTING_DUMP hook (wasim-routing-dump.patch).

<dir>/<point>.layout lists the records of one call ("name count"), <dir>/<point>.f64 holds the calls one
after the other as raw float64. Points: routing_setup (once), routing_in and routing_out (each call of
abflussmodellklasse::run).
"""
import os

import numpy as np


class RoutingDump:
    def __init__(self, directory, point):
        with open(os.path.join(directory, point + ".layout")) as f:
            self.layout = [(name, int(n)) for name, n in (line.split() for line in f if line.strip())]
        self.size = sum(n for _, n in self.layout)
        self.data = np.memmap(os.path.join(directory, point + ".f64"), dtype=np.float64, mode="r")
        if self.data.size % self.size:
            raise ValueError(f"{point}: file size is not a multiple of the record size")
        self.calls = self.data.size // self.size
        self.offset = {}
        pos = 0
        for name, n in self.layout:
            self.offset[name] = (pos, n)
            pos += n

    def __len__(self):
        return self.calls

    def get(self, name, call=0):
        pos, n = self.offset[name]
        start = call * self.size + pos
        return np.array(self.data[start:start + n])

    def names(self, prefix=""):
        return [name for name, _ in self.layout if name.startswith(prefix)]
