"""Read WaSiM's routing description from a (preprocessed) control file into wasim-dace's routing dataclasses.

Use the control file WaSiM writes to its run directory after preprocessing, `control.wasim.intern`
(variables substituted, comments removed). The parsing follows WaSiM 11.01.01 (src/abfluss.cc):
abflussmodellklasse's constructor for the [routing_model] section (a routing description per line
group starting with "TG", elements found by searching "OL", "SUMTRIB", "ZL", "AL", "SP" in the joined
group), SP_class/AL_class::ReadReservoirRule/ReadAbstractionRule for the rule sections (the first
section of a name counts, as WaSiM's findeabschnitt finds it first).
"""
import re

import numpy as np
from wasim_gt.processes import routing_tables as rt

KEYS = ("TG", "OL", "SUMTRIB", "ZL", "AL", "SP")
_NUM = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_INT = re.compile(r"\s*([+-]?\d+)")
_TOK = re.compile(r"\s*(\S+)")


class Stream:
    """Just enough of std::stringstream: a position and >> for double, int and words."""

    def __init__(self, text):
        self.s, self.pos = text, 0

    def seek_after(self, key):
        p = self.s.find(key, self.pos)
        if p < 0:
            raise ValueError(f"{key!r} not found in {self.s!r}")
        self.pos = p + len(key)

    def num(self, rx=_NUM, conv=float):
        m = rx.match(self.s, self.pos)
        if not m:
            raise ValueError(f"no number at {self.s[self.pos:self.pos + 30]!r}")
        self.pos = m.end()
        return conv(m.group(1))

    def int(self):
        return self.num(_INT, int)

    def word(self):
        m = _TOK.match(self.s, self.pos)
        self.pos = m.end()
        return m.group(1)

    def eq_num(self):
        self.seek_after("=")
        return self.num()


def section(lines, name):
    """Index of the line after the first line whose first token is [name]."""
    for i, line in enumerate(lines):
        tok = line.split()
        if tok and tok[0] == f"[{name}]":
            return i + 1
    raise KeyError(name)


def _channel(st, **kw):
    vals = [st.eq_num() for _ in range(10)]
    return rt.Channel(*vals, **kw)


def parse_block(block):
    st = Stream(block)
    st.seek_after("TG")
    code = st.int()
    area = st.eq_num()
    rel = st.eq_num()
    sb = rt.Subbasin(code, area, rel)
    while block.find("OL", st.pos) >= 0:
        st.seek_after("OL")
        up = st.int()
        sb.channels.append(_channel(st, upstream=up))
    st.pos = 0
    while block.find("SUMTRIB", st.pos) >= 0:
        st.seek_after("SUMTRIB")
        up = st.int()
        rest = block[st.pos:]
        cut = min([p for p in (rest.find(c) for c in " (k") if p >= 0], default=len(rest))
        s = rest[:cut]
        members = []
        while "&" in s:
            s = s[s.find("&") + 1:]
            members.append(int(_INT.match(s).group(1)))
        sb.channels.append(_channel(st, upstream=up, members=tuple(members)))
    st.pos = 0
    while block.find("ZL", st.pos) >= 0:
        st.seek_after("ZL")
        num = st.int()
        st.seek_after("=")
        mode = st.word()
        value = 0.0
        if mode == "constant":
            value = st.num()
        elif mode == "extern":
            st.word()
            st.int()
            st.int()
        elif mode != "intern":
            raise ValueError(f"inflow mode {mode}")
        sb.channels.append(_channel(st, inflow=num, inflow_mode=mode, inflow_value=value))
    st.pos = 0
    while block.find("AL", st.pos) >= 0:
        st.seek_after("AL")
        num = st.int()
        st.seek_after("=")
        mode = st.word()
        if ")" in mode:
            mode = mode[:mode.index(")")]
        al = rt.Abstraction(num, mode)
        if mode in ("intern", "extern"):
            if mode == "extern":
                st.word()
            al.threshold, al.fraction, al.maximum = st.num(), st.num(), st.num()
            unit = st.word()
            if ")" in unit:
                unit = unit[:unit.index(")")]
            al.unit_m3s = any(u in unit for u in ("m^3/s", "m3/s", "m3s"))
        sb.abstractions.append(al)
    st.pos = 0
    while block.find("SP", st.pos) >= 0:
        st.seek_after("SP")
        num = st.int()
        st.seek_after("file")
        st.seek_after("=")
        st.word()
        st.seek_after("V0")
        v0 = st.eq_num()
        if "A0" in block or "W0" in block or "UseRunningAvg" in block:
            raise NotImplementedError("A0, W0, UseRunningAvg")
        st.seek_after("C0")
        st.seek_after("=")
        for _ in range(9):
            st.num()
        dtmin = 60.0
        if "dTmin" in block:
            st.seek_after("dTmin")
            dtmin = st.eq_num()
        sb.reservoirs.append(rt.Reservoir(num, v0, rule=None, dt_min=dtmin))
    return sb


def _floats(tokens, n, conv=float):
    out = []
    for i in range(n):
        try:
            out.append(conv(_NUM.match(tokens[i]).group(1)) if conv is float else conv(tokens[i]))
        except (IndexError, AttributeError, ValueError):
            out.append(0)
    return out


def parse_rule(lines, name):
    i = section(lines, name)
    head = lines[i].split()
    tabmax = int(_INT.match(head[0]).group(1))
    if tabmax <= 0:
        raise NotImplementedError(f"{name}: HydraulicConnection")
    rulecnt = len(head)
    if rulecnt > 1:
        rulecnt -= 1
    days = [1] if rulecnt == 1 else [int(t) for t in head[1:1 + rulecnt]]
    index, values = [], [[] for _ in range(rulecnt)]
    for t in range(tabmax):
        tok = lines[i + 1 + t].split()
        v = _floats(tok, rulecnt + 1)
        index.append(v[0])
        for r in range(rulecnt):
            values[r].append(v[r + 1])
    opts = {}
    k = i + 1 + tabmax
    while k < len(lines):
        s = lines[k]
        if "[" in s or s in ("\n", ""):
            break
        for key in ("TargetCap", "WeekDays", "start_hour", "stop_hour", "start2_hour", "stop2_hour", "TableIndex"):
            if key in s:
                rest = s[s.find("=") + 1:]
                if key == "TableIndex":
                    opts["reverse"] = "Flow" in rest
                else:
                    conv = float if key == "TargetCap" else int
                    opts[key] = tuple(_floats(rest.split(), rulecnt, conv))
                break
        k += 1
    return rt.Rule(index=tuple(index), values=tuple(tuple(v) for v in values), days=tuple(days),
                   target_capacity=opts.get("TargetCap"), weekdays=opts.get("WeekDays"),
                   start_hour=opts.get("start_hour"), stop_hour=opts.get("stop_hour"),
                   start_hour2=opts.get("start2_hour"), stop_hour2=opts.get("stop2_hour"),
                   reverse=opts.get("reverse", False))


def read_routing(control_file):
    """Returns (subbasins, dict of the section's header values, lines of the control file)."""
    with open(control_file, encoding="latin-1") as f:
        lines = f.readlines()
    i = section(lines, "routing_model")
    head = lines[i + 2].split()
    hdr = {"spec_min": float(head[0]), "spec_max": float(head[1]), "intervals": float(head[2]), "subint": int(head[3]),
           "do_it": int(lines[i].split()[0]), "time_step": float(lines[i + 1].split()[0])}
    ncol = int(lines[i + 5].split()[0])
    k = i + 6 + ncol + 1
    groups = []
    while k < len(lines) and any(key in lines[k] for key in KEYS):
        if "TG" in lines[k]:
            groups.append(lines[k].rstrip("\n"))
        else:
            groups[-1] += lines[k].rstrip("\n")
        k += 1
    subbasins = [parse_block(g) for g in groups]
    for sb in subbasins:
        for res in sb.reservoirs:
            res.rule = parse_rule(lines, f"abstraction_rule_reservoir_{res.code}")
        for al in sb.abstractions:
            if al.mode in ("intern_with_rule", "extern_with_rule", "intern_with_rule_from_reservoir"):
                al.rule = parse_rule(lines, f"abstraction_rule_abstraction_{al.code}")
    j = section(lines, "output_interval")
    hdr["runoff_measure"] = int(lines[j + 2].split()[0])
    return subbasins, hdr, lines


def read_ascii_grid(path):
    """ESRI ASCII grid as WaSiM reads it (rows from the north)."""
    with open(path, encoding="latin-1") as f:
        head = [f.readline().split() for _ in range(6)]
        data = np.array(f.read().split(), dtype=np.float64)
    h = {k.lower(): float(v) for k, v in head}
    return data.reshape(int(h["nrows"]), int(h["ncols"])), h


def grid_path(lines, identification):
    """File name of a standard grid with the given identification."""
    for line in lines:
        tok = line.split()
        if len(tok) >= 2 and tok[1] == identification:
            return tok[0]
    raise KeyError(identification)
