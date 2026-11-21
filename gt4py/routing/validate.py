"""Check wasim-dace's routing (lakes, reservoirs, abstractions, inflows) against WaSiM, every step.

Reference: a WaSiM 11.01.01 run (wasim-eth eab8916, the reference binary of the other ports) built
with the routing dump hook (`wasim-routing-dump.patch` in this directory: `git apply` in wasim-eth,
then `make`; it only writes files, qgko.dat and the other outputs stay byte-identical) and run with
WASIM_ROUTING_DUMP=<dump dir>:

    WASIM_ROUTING_DUMP=<dump> run_case.py --binary <patched wasim> --template golden/doubs/control.wasim.in \\
        --case-dir <Doubs> --out <run>

Rhone: template golden/rhone/control.wasim.in (with the merged station files, golden/README.md).

The hook writes the full routing state at the start and the end of every call of
abflussmodellklasse::run (routing_in / routing_out) and the tables built at initialisation
(routing_setup), all as float64 (the text outputs have 5-6 digits only).

What is compared, with tolerance exactly 0:
1. the tables built by `routing_tables.build` from the control file (`<run>/control.wasim.intern`)
   against routing_setup: relative areas and weights, w-q tables (translation steps, flood-plain
   fractions, velocities), retention constants, abstraction parameters, lake level/volume tables;
2. one chained run from WaSiM's state at the first call: the inputs of each step are taken from
   routing_in (runoff of the soil model, river infiltration, POND/ETR/EI/q_down/baseflow/precipitation
   on lake cells), the state is the port's own. After each step the whole state is compared with
   routing_out: Qges (all subbasins, inflows and abstractions), Q12ges, the flow arrays, per channel
   last_index / ueberrollt / qvzuq / Qaus, per reservoir volume / outflow / level / area / balance
   terms, abstractions from reservoirs, the river-flow table and POND on lake cells;
3. the routed discharge per subbasin against qgko.dat, and reservoir balances against the spv_*
   files, at the printed precision.

usage: validate.py --run RUN --dump DUMP [--steps N]

Run from a wasim-dace checkout: NPY_DISABLE_CPU_FEATURES=X86_V4 uv run python <this file> --run ... --dump ...
The time printed is the compiled program's (prepared arguments, `routing.RoutingStepper`), to compare
with WaSiM's `after_post_run -> after_routing` segment of WASIM_TIMING.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wasim_gt.cells import CellLayout
from wasim_gt.processes import routing as rp
from wasim_gt.processes import routing_tables as rt

from control import read_routing
from dump import RoutingDump


def maxdiff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes {a.shape} {b.shape}")
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(a - b))
    return float(np.nanmax(d)) if d.size else 0.0 if not np.isnan(d).any() else math.inf


def config_from_run(run, setup):
    subbasins, hdr, lines = read_routing(os.path.join(run, "control.wasim.intern"))
    sizes = setup.get("sizes")
    nrows, ncols, cellsize, nodata = setup.get("grid")
    nrows, ncols = int(nrows), int(ncols)
    zone_grid = setup.get("zone_grid").reshape(nrows, ncols)
    layout = CellLayout.from_mask(zone_grid != nodata)
    codes, counts = np.unique(zone_grid[zone_grid != nodata].astype(int), return_counts=True)
    if not np.array_equal(codes, setup.get("zone_code")[1:].astype(int)):
        raise ValueError("zone codes of the grid differ from WaSiM's zones")
    lakes = {}
    for name in setup.names():
        if name.endswith("_cells"):
            head = setup.get(name.replace("_cells", "_head"))
            cells = setup.get(name).reshape(-1, 4)  # row, column, DEM, MAXPOND (both as the lake model set them)
            i, j = cells[:, 0].astype(int) - 1, cells[:, 1].astype(int) - 1
            idx = layout.index[i, j]
            if np.any(idx < 0):
                raise ValueError(f"lake {int(head[0])}: cells outside the model")
            lakes[int(head[0])] = rt.LakeCells(cell=idx, dem=cells[:, 2], max_pond=cells[:, 3], in_scan=(j + 1) < ncols)
    cfg = rt.RoutingConfig(
        subbasins=subbasins, zone_codes=codes, zone_cells=counts,
        cellsize=cellsize, spec_min=hdr["spec_min"], spec_max=hdr["spec_max"], intervals=hdr["intervals"],
        subint=hdr["subint"], time_step=sizes[12], timestep_factor=int(sizes[10]), runoff_measure=int(sizes[11]),
        lake_active=bool(sizes[7]), lakes=lakes, riverflow=bool(sizes[6]))
    if sizes[5] or sizes[8] or sizes[9]:
        raise NotImplementedError("irrigation, observed inflows or tracers in this run")
    return cfg, layout, lines


def check_tables(t, setup):
    """Tables against routing_setup; returns the largest difference."""
    A = t.arrays
    worst = 0.0
    rows = []

    def cmp(name, a, b):
        nonlocal worst
        d = maxdiff(a, b)
        worst = max(worst, d)
        if d != 0:
            rows.append((name, d))

    cmp("abfrel0", t.abfrel0, setup.get("abfrel0"))
    cmp("abfrel1", t.abfrel1, setup.get("abfrel1"))
    for r, el in enumerate(t.meta["elements"], start=1):
        head = setup.get(f"r{r}_head")
        cmp(f"r{r} TG", el["sb"].code, head[0])
        cmp(f"r{r} TgAe", A["el_tgae"][r - 1], head[1])
        cmp(f"r{r} gew", el["gew"], setup.get(f"r{r}_gew"))
        for i in range(1, A["el_c1"][r - 1] - A["el_c0"][r - 1] + 1):
            c = A["el_c0"][r - 1] + i - 1
            h = setup.get(f"r{r}_c{i}_head")
            nb = A["ch_nb"][c]
            cmp(f"r{r}c{i} n,n12t,kh,kv,AE,dq", [A["ch_n"][c], t.meta["ch_rows"][c]["tab"]["n12t"], A["ch_kh"][c],
                                                  A["ch_kv"][c], A["ch_ae"][c], A["ch_dq"][c]], h[:6])
            cmp(f"r{r}c{i} delta", A["ch_delta"][c, 1:nb + 1], setup.get(f"r{r}_c{i}_delta"))
            cmp(f"r{r}c{i} QvzuQ", A["ch_qvq"][c, 1:nb + 1], setup.get(f"r{r}_c{i}_QvzuQ"))
            cmp(f"r{r}c{i} v", A["ch_v"][c, 1:nb + 1], setup.get(f"r{r}_c{i}_v"))
        for a in range(1, A["el_a1"][r - 1] - A["el_a0"][r - 1] + 1):
            k = A["el_a0"][r - 1] + a - 1
            h = setup.get(f"r{r}_a{a}_head")
            cmp(f"r{r}a{a}", [A["al_art"][k], A["al_code"][k], A["al_schw"][k], A["al_ant"][k], A["al_max"][k],
                              A["al_tgae"][k], A["al_tgaerel"][k], A["al_zlt"][k], A["al_zlf"][k], A["al_srcae"][k]],
                h[:10] if A["al_zlt"][k] > 0 else np.concatenate([h[:8], [0, 0.0]]))  # unset for extern
        for s in range(1, A["el_s1"][r - 1] - A["el_s0"][r - 1] + 1):
            k = A["el_s0"][r - 1] + s - 1
            h = setup.get(f"r{r}_s{s}_head")
            nw = A["sp_nw"][k]
            cmp(f"r{r}s{s} code,dTmin,nw", [t.reservoir_codes[k], A["sp_dtmin"][k], nw], [h[0], h[1], h[4]])
            if h[3]:
                cmp(f"r{r}s{s} level", A["sp_level"][k, :nw], setup.get(f"r{r}_s{s}_level"))
                cmp(f"r{r}s{s} volume", A["sp_volume"][k, :nw], setup.get(f"r{r}_s{s}_volume"))
    return worst, rows


class Mapping:
    """Positions of WaSiM's per-routing records in the port's arrays."""

    def __init__(self, t):
        A = t.arrays
        self.channels, self.reservoirs, self.abstractions = [], [], []
        for r in range(len(A["el_tg"])):
            for i in range(1, A["el_c1"][r] - A["el_c0"][r] + 1):
                self.channels.append((f"r{r + 1}c{i}_", A["el_c0"][r] + i - 1))
            for s in range(1, A["el_s1"][r] - A["el_s0"][r] + 1):
                self.reservoirs.append((f"r{r + 1}s{s}_", A["el_s0"][r] + s - 1))
            for a in range(1, A["el_a1"][r] - A["el_a0"][r] + 1):
                self.abstractions.append((f"r{r + 1}a{a}_", A["el_a0"][r] + a - 1))


def state_from_dump(t, d, call, mp):
    st = rp.State.empty(t)
    nq = t.sizes["NQ"]
    st.qges[:] = d.get("qges", call)[:nq]
    subint = t.meta["subint"]
    st.q12[1:subint + 1, :] = d.get("q12ges", call).reshape(subint, -1)
    A = t.arrays
    for prefix, c in mp.channels:
        n = A["ch_n"][c]
        geb = A["ch_geb"][c]
        st.last_index[c, 1:n + 1] = d.get(prefix + "last_index", call)
        st.ueb[c, 1:n + 1] = d.get(prefix + "ueberrollt", call)
        st.qvzuq[c, 1:n + 1] = d.get(prefix + "qvzuq", call)
        st.qaus[c, 1:n + 1, 1:subint + 1] = d.get(prefix + "qaus", call).reshape(n, subint)
        fl = d.get(prefix + "fluss", call)
        st.fluss[geb, 1:fl.size + 1] = fl
    for prefix, s in mp.reservoirs:
        v = d.get(prefix + "state", call)
        st.sp_vakt[s], st.sp_abakt[s], st.sp_wlevel[s], st.sp_area[s], st.sp_dpond[s], st.sp_vold[s] = v[:6]
    for prefix, a in mp.abstractions:
        st.al_abakt[a] = d.get(prefix + "state", call)[0]
    rf = d.get("riverflow", call).reshape(-1, 6)
    z = rf.shape[0]
    st.rf_flow_mm[1:z + 1], st.rf_flowm3s[1:z + 1], st.rf_speed[1:z + 1] = rf[:, 0], rf[:, 1], rf[:, 2]
    st.infilt[1:z + 1], st.infilt_frac[1:z + 1] = rf[:, 4], rf[:, 5]
    return st


def lake_inputs(t, d, call, mp, n):
    """Model-cell arrays of POND, ETR, EI, q_downext, baseflow and PrecOnLakes, set on lake cells."""
    grids = np.zeros((6, n))
    A = t.arrays
    for prefix, s in mp.reservoirs:
        v = d.get(prefix + "lake", call).reshape(-1, 6)
        cells = A["lk_cell"][A["sp_l0"][s]:A["sp_l1"][s]]
        grids[:, cells] = v.T
    return grids


def compare_state(t, st, d, call, mp, pond):
    A = t.arrays
    subint = t.meta["subint"]
    out = {}

    def put(name, value):
        out[name] = max(out.get(name, 0.0), value)

    put("qges", maxdiff(st.qges, d.get("qges", call)))
    put("q12ges", maxdiff(st.q12[1:subint + 1, :], d.get("q12ges", call).reshape(subint, -1)))
    for prefix, c in mp.channels:
        n = A["ch_n"][c]
        geb = A["ch_geb"][c]
        put("last_index", maxdiff(st.last_index[c, 1:n + 1], d.get(prefix + "last_index", call)))
        put("ueberrollt", maxdiff(st.ueb[c, 1:n + 1], d.get(prefix + "ueberrollt", call)))
        put("qvzuq", maxdiff(st.qvzuq[c, 1:n + 1], d.get(prefix + "qvzuq", call)))
        put("Qaus", maxdiff(st.qaus[c, 1:n + 1, 1:subint + 1].ravel(), d.get(prefix + "qaus", call)))
        fl = d.get(prefix + "fluss", call)
        put("fluss", maxdiff(st.fluss[geb, 1:fl.size + 1], fl))
    for prefix, s in mp.reservoirs:
        v = d.get(prefix + "state", call)
        put("reservoir V", maxdiff(st.sp_vakt[s], v[0]))
        put("reservoir Qout", maxdiff(st.sp_abakt[s], v[1]))
        put("reservoir level", maxdiff(st.sp_wlevel[s], v[2]))
        put("reservoir area", maxdiff(st.sp_area[s], v[3]))
        lake = d.get(prefix + "lake", call).reshape(-1, 6)
        cells = A["lk_cell"][A["sp_l0"][s]:A["sp_l1"][s]]
        put("POND (lake cells)", maxdiff(pond[cells], lake[:, 0]))
    for prefix, a in mp.abstractions:
        put("abstraction", maxdiff(st.al_abakt[a], d.get(prefix + "state", call)[0]))
    rf = d.get("riverflow", call).reshape(-1, 6)
    z = rf.shape[0]
    put("riverflow", maxdiff(np.stack([st.rf_flow_mm[1:z + 1], st.rf_flowm3s[1:z + 1], st.rf_speed[1:z + 1],
                                       st.infilt[1:z + 1], st.infilt_frac[1:z + 1]]),
                             np.stack([rf[:, 0], rf[:, 1], rf[:, 2], rf[:, 4], rf[:, 5]])))
    return out


def compare_balances(res, d, call, mp):
    out = 0.0
    for prefix, s in mp.reservoirs:
        b = d.get(prefix + "balance", call)  # QIN QOUT IRRIG QGW ETR PREC DV ABSTR ERR W
        mine = res.reservoirs[s]
        out = max(out, maxdiff([mine[3], mine[4], mine[5], mine[6], mine[7], mine[8], mine[9], mine[10], mine[11], mine[0]],
                               b))
    return out


def text_outputs(run):
    """Reservoir (spv) and abstraction files of the run, by code: {code: rows of tokens}."""
    res, abst = {}, {}
    for name in sorted(os.listdir(run)):
        path = os.path.join(run, name)
        if not name.endswith(".dat") or not os.path.isfile(path):
            continue
        with open(path, encoding="latin-1") as f:
            head = [f.readline() for _ in range(3)]
            if head[1].startswith("reservoir_"):
                res[int(head[1].split("_")[1])] = [line.split() for line in f]
            elif head[1].startswith("abstraction_"):
                abst[int(head[1].split("_")[1])] = [line.split() for line in f]
    return res, abst


def spv_row(out, abakt, inv):
    """The 30 values WaSiM writes per reservoir and step (SP_class::WriteSpeicherinhalt)."""
    w, area, v, qin, qout, irr, qgw, etr, prec, dv, ab, err = out
    mm = lambda x: x / area * 1000.0 if area > 0 else 0.0
    m3s = lambda x: x / (inv * 60.0)
    return [w, area, v, qin, mm(qin), m3s(qin), qout, mm(qout), abakt, irr, mm(irr), m3s(irr), qgw, mm(qgw), m3s(qgw),
            etr, mm(etr), m3s(etr), prec, mm(prec), m3s(prec), dv, mm(dv), m3s(dv), ab, mm(ab), m3s(ab), err, mm(err),
            m3s(err)]


def read_table(path, skip):
    rows = []
    with open(path) as f:
        for k, line in enumerate(f):
            if k < skip:
                continue
            rows.append(line.split())
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="WaSiM run directory (control.wasim.intern, qgko.dat, spv_*)")
    ap.add_argument("--dump", required=True, help="WASIM_ROUTING_DUMP directory")
    ap.add_argument("--steps", type=int, default=0, help="number of calls (default: all)")
    args = ap.parse_args()

    setup = RoutingDump(args.dump, "routing_setup")
    din = RoutingDump(args.dump, "routing_in")
    dout = RoutingDump(args.dump, "routing_out")
    cfg, layout, lines = config_from_run(args.run, setup)
    t = rt.build(cfg)
    worst, rows = check_tables(t, setup)
    print(f"tables: {len(t.meta['ch_rows'])} channels, {len(t.meta['elements'])} subbasins, "
          f"{len(t.meta['al_rows'])} abstractions, {len(t.meta['sp_rows'])} reservoirs, {t.meta['n_lake']} lake cells: "
          f"max diff {worst}")
    for name, dd in rows[:20]:
        print(f"  {name}: {dd}")
    mp = Mapping(t)
    n = layout.n
    steps = args.steps or len(din)
    st = state_from_dump(t, din, 0, mp)
    # inflows from files: WaSiM reads the first value column of each ZL file at the date of the step
    zl_files = {}
    A = t.arrays
    for c, row in enumerate(t.meta["ch_rows"]):
        if row["kind"] == rt.INFLOW and row["zlart"] == rt.ZL_EXTERN:
            zl_files[c] = row
    zl_series = inflow_series(t, lines, zl_files)

    stepper = rp.RoutingStepper(t, st, n)
    spv, abst = text_outputs(args.run)
    text_bad = text_n = 0
    qgko = read_table(os.path.join(args.run, "qgko.dat"), 3)
    worst_step = {}
    worst_q = 0.0
    worst_bal = 0.0
    times = []
    for call in range(steps):
        date = din.get("date", call).astype(int)
        year, month, day, hour, _minute, actsub, numsub = date
        st_in = state_from_dump(t, din, call, mp)
        # couplings set by the other processes between the calls: river infiltration (soil model)
        st.infilt[:] = st_in.infilt
        st.infilt_frac[:] = st_in.infilt_frac
        grids = lake_inputs(t, din, call, mp, n)
        qs = din.get("qges_soil", call)
        inp = stepper.inputs
        inp["qges_soil"][1:qs.size + 1] = qs
        for k, name in enumerate(("pond", "etr", "ei", "qu", "qb", "prec")):
            inp[name][:] = grids[k]
        for c in zl_files:
            inp["zl_value"][c] = zl_series[c].get((year, month, day, hour), np.nan)
        res = stepper(year, month, day, hour, actsub, numsub)
        times.append(res.seconds)
        diffs = compare_state(t, st, dout, call, mp, inp["pond"])
        for k, v in diffs.items():
            worst_step[k] = max(worst_step.get(k, 0.0), v)
        worst_bal = max(worst_bal, compare_balances(res, dout, call, mp))
        # qgko: 5 decimals
        row = qgko[call]
        if [int(x) for x in row[:4]] != [year, month, day, hour]:
            raise ValueError(f"qgko date {row[:4]} != {date[:4]}")
        printed = np.array([float(x) for x in row[4:4 + t.zonmax]])
        mine = np.array([float(f"{x:.5f}") for x in res.q_out[1:t.zonmax + 1]])
        worst_q = max(worst_q, maxdiff(printed, mine))
        # reservoir and abstraction files: %lg (6 significant digits)
        for k, code in enumerate(t.reservoir_codes):
            mine = [f"{x:g}" for x in spv_row(res.reservoirs[k], st.sp_abakt[k], t.meta["inv"])]
            text_n += 1
            text_bad += mine != spv[code][call][4:]
        for k, code in enumerate(t.abstraction_codes):
            if code in abst and A["el_a0"].size and not np.isnan(res.abstractions[k]):
                text_n += 1
                text_bad += f"{res.abstractions[k]:g}" != abst[code][call][4]
        if call == 0 or any(v != 0 for v in diffs.values()):
            bad = {k: v for k, v in diffs.items() if v != 0}
            if bad or call == 0:
                print(f"call {call} {year}-{month:02d}-{day:02d} {hour:02d}: {bad if bad else 'all 0'}")
    print(f"\n{steps} calls, state compared with routing_out after every call (max abs diff):")
    for k, v in worst_step.items():
        print(f"  {k:20s} {v:.3e}")
    print(f"  {'reservoir balances':20s} {worst_bal:.3e}")
    print(f"qgko.dat (5 decimals): max diff {worst_q:.1e}")
    print(f"spv_* and abstraction files (%lg): {text_bad} of {text_n} rows differ")
    tm = np.array(times[1:]) * 1000
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"routing program: {np.median(tm):.4f} ms median, {tm.mean():.4f} ms mean per call (warm, compiled SDFG "
          f"called with prepared arguments, without gather; OMP_NUM_THREADS={threads})")


def inflow_series(t, lines, zl_files):
    """Values of the external inflow files: the first value column at each date (WaSiM reads
    GetActValue(0) whatever column the control file names)."""
    out = {}
    for c, row in zl_files.items():
        number = row["geb"] - t.zonmax
        name = None
        for line in lines:
            if "ZL" in line and "extern" in line:
                toks = line.replace("(", " ").split()
                for a, tok in enumerate(toks[:-1]):
                    if tok == "ZL" and toks[a + 1].isdigit() and int(toks[a + 1]) == number:
                        name = toks[toks.index("extern") + 1]
        if name is None:
            raise KeyError(f"ZL {number}")
        series = {}
        with open(name, encoding="latin-1") as f:
            for line in f:
                tok = line.split()
                if len(tok) >= 5 and tok[0].isdigit():
                    series[(int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3]))] = float(tok[4])
        out[c] = series
    return out


if __name__ == "__main__":
    main()
