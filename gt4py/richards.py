#!/usr/bin/env python3
"""Validate wasim-gt's Richards column (wasim_gt.soil.richards, WaSiM unsatzon method 3) against WaSiM dumps.

The inputs are the grids WaSiM had before the soil model's call (dump point after_siltingup; between
it and the call only the pre-runs of the soil, surface-routing, ET and groundwater modules run, and
with one module sub-step they change no cell grid: the soil pre_run shifts the travel-time zone
registers, see gt4py/richards.md §2), the results are compared with the dump after_unsatzon, for
every step and every active cell. Each step starts from WaSiM's own state, so a difference cannot
propagate. Option set: Doubs and Rhone golden cases (gt4py/richards.md §1). Under surface
interaction 2 (Doubs) the cells with a river link are computed in WaSiM's separate ordered row with the
river-flow table of the routing (#8): their order comes from the elevation model as read (the file
named in [elevation_model] of CONTROL_INTERN, an ASCII grid), the table from the river-flow table dump
below, and the table after the module and after post_run is compared as well.

Dump grids (both points unless marked):
  theta, hydraulic_heads, q_down, groundwater_level, groundwater_distance, Theta_in_GWLEVEL, baseflow,
  ponding_storage, real_evaporation, real_transpiration0..2;
  before only: geodetic_heads, throughfall0 (interception after snow) or snowcover_outflow (before), potential_evaporation, pot_transpiration0..2,
  snowstorage_solid, soil_types, zonegrid_soilmodel, landuse1..2, vegetation_coverage_degree1..2,
  root_depth1..2, lake_codes, max_ponding_storage, river_depth, river_width, river_links, kolmation,
  tangens_beta, GlacierizedCells (if glaciers);
  after only: SB_1_grid, SB_2_grid, MeanThetaInRootZone, ROOTgrid, macropore_infiltration,
  number_of_substeps, groundwater_recharge, Capillar_Uprise, Percolation, directflow, interflow,
  infiltration_excess, total_infiltration, river_infiltration, river_exfiltration, gw_storage_coeff_1,
  ETR2ETP.
Points: after_siltingup,after_unsatzon. Template: golden/<case>/control.wasim.in with the end date
moved (48 steps: $endday = 3).
River-flow table (surface interaction 2 only): WaSiM 11.01.01 (eab8916) with the table dump hook
gt4py/richards_rft_dump.patch applied to src/unsatzon.cc (env WASIM_DUMP_RFT=<DUMP_DIR>, same run):
rft_run_start.f64 / rft_run_end.f64 / rft_post_run.f64 (per call: zonmax+1 records of flowm3s, speed,
infilt_amount, infilt_fraction) and rft_order.txt (WaSiM's river-link row, checked against ours).

The port is a DaCe program on the model cells (wasim_gt.cells): grids are gathered to cells, the
program runs, and its outputs are compared with the dump on the same cells. --python runs the same
code as plain Python (slow; with --rows to restrict it to some grid rows), to tell a DaCe defect
from a porting error.

usage: richards.py DUMP_DIR CONTROL_INTERN [--steps N] [--fix] [--python --rows A:B] [--cellsize M]
"""
import argparse
import re
import time

import numpy as np

from wasim_gt.cells import CellLayout
from wasim_gt.soil import richards as rz

from wasim_dump import Dump

NODATA = -9999.0
PRE, POST = "after_siltingup", "after_unsatzon"


def section(text, name):
    """Non-empty lines of a control-file section, comments removed."""
    body = text.split(f"[{name}]", 1)[1].split("\n[", 1)[0]
    lines = [ln.split("#", 1)[0].strip() for ln in body.splitlines()]
    return [ln for ln in lines if ln]


def records(text, name):
    """[(code, body)] of a table section ("code name { ... }"), in file order."""
    body = text.split(f"[{name}]", 1)[1].split("\n[", 1)[0]
    body = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
    return [(int(m.group(1)), m.group(2)) for m in re.finditer(r"^\s*(\d+)\s+\S+\s*\{(.*?)\}", body, re.M | re.S)]


def single(body, key, default=None, first=True):
    """WaSiM's stripSingleParameter / atof: first number after '=' following the first occurrence."""
    i = body.find(key) if first else body.rfind(key)
    if i < 0:
        return default
    buf = body[i:]
    buf = buf[buf.find("=") + 1:buf.find(";")]
    m = re.match(r"\s*([-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?)", buf)
    return float(m.group(1)) if m else 0.0


def multi(body, key):
    """SOILtabletype::stripParameters: the values after the last occurrence of key."""
    i = body.rfind(key)
    if i < 0:
        return None
    buf = body[i:]
    buf = buf[buf.find("=") + 1:buf.find(";")]
    return [float(x) for x in re.split(r"[\t ;,\n]+", buf.strip()) if x]


def soil_types(text):
    out = []
    for code, body in records(text, "soil_table"):
        if "MultipleHorizons" not in body:
            raise SystemExit(f"soil type {code}: only method MultipleHorizons is supported")
        has = lambda k: body.find(k) >= 0
        out.append(rz.SoilType(
            layers=[int(x) for x in multi(body, "layers")], ksat=multi(body, "ksat"),
            theta_sat=multi(body, "theta_sat"), theta_res=multi(body, "theta_res"), alpha=multi(body, "alpha"),
            par_n=multi(body, "Par_n"), thickness=multi(body, "thickness"),
            par_tau=multi(body, "Par_tau") if has("Par_tau") else None,
            par_m=multi(body, "Par_m") if has("Par_m") else None,
            k_recession=multi(body, "k_recession") if has("k_recession") else None,
            thickness_scaling=single(body, "ThicknessScaling", 1.0),
            max_interflow_depth=single(body, "MaxInterflowDepth", 2.0),
            evap_max_depth=single(body, "EvapMaxDepth", None),
            maxratio=single(body, "maxratio", 1000.0), pmacro_thresh=single(body, "PMacroThresh", 1000.0),
            macro_capacity=single(body, "MacroCapacity", 0.0), capacity_redu=single(body, "CapacityRedu", 1.0),
            macro_depth=single(body, "MacroDepth", 1.0)))
    return out


def landuse_parameters(text):
    """{position: (RootDist, HReduDry, TReduWet, LimitReduWet)}; WaSiM replaces the land-use codes in
    the grids by the record position (defltuzr.cc:1825)."""
    out = {}
    for pos, (code, body) in enumerate(records(text, "landuse_table"), start=1):
        out[pos] = (single(body, "RootDist"), single(body, "HReduDry"), single(body, "TReduWet"),
                    single(body, "LimitReduWet", 0.0))
    return out


def unsatzon_options(text):
    sec = section(text, "unsatzon_model")
    if int(sec[2]) != 3:
        raise SystemExit("only soil method 3 is ported")
    if int(sec[5]) != 0 or int(sec[6]) != 0:
        raise SystemExit("drainage and clay layer are not ported")
    nums = lambda ln: [float(x) for x in ln.split()]
    isnum = lambda ln: all(re.fullmatch(r"[-+0-9.eE]+", x) for x in ln.split())
    codes = None
    for i, ln in enumerate(sec):
        if i > 8 and isnum(ln) and len(ln.split()) > 1 and i + 6 < len(sec) and isnum(sec[i + 3]) \
                and len(sec[i + 3].split()) >= len(ln.split()):
            codes = i
            break
    zones = [int(x) for x in nums(sec[codes])]
    nz = len(zones)
    dr = nums(sec[codes + 3])[:nz]
    sdf = nums(sec[codes + 6])[:nz]
    min_step = float(sec[codes + 9].split()[0])
    tuning = section(text, "unsatzon_tuning")
    maxerr = float([t for t in tuning if t.startswith("maxerror")][0].split("=")[1])
    maxit = float([t for t in tuning if t.startswith("max_iterations")][0].split("=")[1])
    lake = int(section(text, "lake_model")[0]) if "[lake_model]" in text else 0
    opts = rz.Options(interval=float(sec[1]), substeps=1, min_step=min_step, max_error=maxerr,
                      max_iterations=maxit, surface_interaction=int(sec[3]), gwflow=1, lake=lake,
                      ponds=int(sec[4]))
    return opts, dict(zip(zones, dr)), dict(zip(zones, sdf))


def ascii_grid(path):
    """ESRI ASCII grid as written for WaSiM (6 header lines), as float64 rows x cols."""
    with open(path, encoding="latin-1") as f:
        head = [next(f).split() for _ in range(6)]
        ncols, nrows = int(head[0][1]), int(head[1][1])
        vals = np.array(f.read().split(), dtype=np.float64)
    return vals[:nrows * ncols].reshape(nrows, ncols)


def river_flow_dump(dump_dir):
    """River-flow table dumps {name: (calls, zonmax+1, 4)} (flowm3s, speed, infilt_amount, infilt_fraction)."""
    import os
    head = open(os.path.join(dump_dir, "rft_order.txt")).readline().split()
    nz = int(head[head.index("zonmax") + 1]) + 1
    out = {}
    for name in ("rft_run_start", "rft_run_end", "rft_post_run"):
        out[name] = np.fromfile(os.path.join(dump_dir, name + ".f64")).reshape(-1, nz, 4)
    return out, nz


def check_link_order(dump_dir, cells, link_cells, lnk):
    """Our river-link row against WaSiM's (rft_order.txt: row col link zarray[link] elevation, 1-based)."""
    import os
    ref = np.loadtxt(os.path.join(dump_dir, "rft_order.txt"), comments="#", ndmin=2)
    mine = np.c_[cells.rows[link_cells] + 1, cells.cols[link_cells] + 1, lnk[link_cells]]
    if mine.shape != (ref.shape[0], 3) or not np.array_equal(mine, ref[:, [0, 1, 3]].astype(np.int64)):
        raise SystemExit("river-link row differs from WaSiM's (rft_order.txt)")
    print(f"river-link row: {len(link_cells)} cells, same order and table indices as WaSiM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("control_intern", help="control.wasim.intern of the run (WaSiM writes it next to the control file)")
    ap.add_argument("--steps", type=int, default=0, help="number of steps (default: all in the dump)")
    ap.add_argument("--first", type=int, default=0, help="first step")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cellsize", type=float, default=500.0)
    ap.add_argument("--fix", action="store_true", help="independent columns (proposed h_temp1 fix); WaSiM "
                    "unfixed carries a row buffer, so a difference here means the carry-over is reached")
    ap.add_argument("--python", action="store_true", help="run the column code as plain Python")
    ap.add_argument("--rows", default=None, help="only grid rows A:B (0-based, end exclusive)")
    ap.add_argument("--show", type=int, default=0, help="print the first N differing cells per grid")
    args = ap.parse_args()

    text = open(args.control_intern, encoding="latin-1").read()
    opts, fd_zone, sr_zone = unsatzon_options(text)
    opts.cellsize = args.cellsize
    opts.fix_htemp1 = 1 if args.fix else 0
    tables = rz.soil_tables(soil_types(text))
    lu = landuse_parameters(text)
    d = Dump(args.dump_dir)
    nv = args.layers
    nl = tables.levels

    zn = d(PRE, "zonegrid_soilmodel")[0]
    mask = zn > NODATA + 1
    if args.rows:
        a, b = (int(x) for x in args.rows.split(":"))
        rows_ok = np.zeros(mask.shape, bool)
        rows_ok[a:b] = True
        mask &= rows_ok
    cells = CellLayout.from_mask(mask)
    n = cells.n
    g = lambda point, name, t, dtype=np.float64: cells.to_cells(d(point, name)[t], dtype)
    st0 = lambda name: g(PRE, name, 0)

    kind = np.full(n, rz.COLUMN, dtype=np.int64)
    if d.has(PRE, "GlacierizedCells"):
        ice = np.clip(st0("GlacierizedCells"), 0.0, 1.0)
        kind[ice == 1.0] = rz.GLACIER
    rivlnk = st0("river_links")
    rftab = rft = None
    lnk = np.zeros(n, dtype=np.int64)
    if opts.surface_interaction > 1:
        dem = cells.to_cells(ascii_grid(section(text, "elevation_model")[0]))
        order, row_start, skip = rz.cell_order(cells.rows, rivlnk, dem, opts.surface_interaction, opts.fix_htemp1)
        kind[skip] = rz.SKIP
        zgrid = d(PRE, "zonegrid_soilmodel")[0]
        lnk = rz.link_index(rivlnk, zgrid[zgrid > NODATA + 1])
        rft, nz = river_flow_dump(args.dump_dir)
        if not args.rows:
            check_link_order(args.dump_dir, cells, order[:row_start[1]], lnk)
    else:
        order, row_start = np.arange(n, dtype=np.int64), rz.rows_of(cells.rows, opts.fix_htemp1)
    nlink = int(row_start[1]) if opts.surface_interaction > 1 else 0
    soil = st0("soil_types").astype(np.int64)
    zcode = st0("zonegrid_soilmodel").astype(np.int64)
    # zones without own parameters take those of the next lower zone (unsatzon.cc:888-1010)
    allz = np.unique(d(PRE, "zonegrid_soilmodel")[0][d(PRE, "zonegrid_soilmodel")[0] > NODATA + 1]).astype(int)
    for table in (fd_zone, sr_zone):
        prev = None
        for z in allz:
            if z in table:
                prev = table[z]
            else:
                table[z] = prev
    luse = [st0(f"landuse{v}") for v in range(1, nv + 1)]
    lu_valid = np.array([(x > NODATA).astype(np.int64) for x in luse])
    lu_par = lambda j: np.array([[lu[int(c)][j] if c > NODATA else 0.0 for c in x] for x in luse])
    static = dict(soil=soil, zone_fd=np.array([fd_zone[z] for z in zcode]),
                  zone_sr=np.array([sr_zone[z] for z in zcode]), lu_valid=lu_valid,
                  lu_formpar=lu_par(0), lu_redudry=lu_par(1), lu_reduwet=lu_par(2), lu_limitwet=lu_par(3),
                  lake=st0("lake_codes"), maxpond=st0("max_ponding_storage"), rivdep=st0("river_depth"),
                  rivwit=st0("river_width"), kolm=st0("kolmation"), tanbeta=st0("tangens_beta"))
    opts.snow = 1 if d.has(PRE, "snowstorage_solid") else 0
    # input of the soil model (setInterceptionBeforeSnow, unsatzon.cc:7916): snow outflow first when
    # interception runs before snow (methods > 10), else throughfall first
    icpt = int(section(text, "interception_model")[2].split()[0])
    prec_grid = "snowcover_outflow" if icpt > 10 else "throughfall0"

    def stack(point, name, t):
        a = np.zeros((n, nl + 2))
        a[:, :nl + 1] = g(point, name, t)
        return a

    hgeo = stack(PRE, "geodetic_heads", 0)
    scalar_in = {rz.O_GWLEVEL: "groundwater_level", rz.O_GWSTAND: "groundwater_distance",
                 rz.O_GWTHETA: "Theta_in_GWLEVEL", rz.O_DELTAGW: "baseflow", rz.O_POND: "ponding_storage",
                 rz.O_EVAR: "real_evaporation"}
    compare = dict(scalar_in)
    compare.update({rz.O_QD: "directflow", rz.O_QI: "interflow", rz.O_GWEX: "river_exfiltration",
                    rz.O_GWIN: "river_infiltration", rz.O_MACRO: "macropore_infiltration",
                    rz.O_GWN: "groundwater_recharge", rz.O_PERCOL: "Percolation", rz.O_UPRISE: "Capillar_Uprise",
                    rz.O_INFEX: "infiltration_excess", rz.O_INFILT: "total_infiltration",
                    rz.O_SUBSTEPS: "number_of_substeps", rz.O_SB1: "SB_1_grid", rz.O_SB2: "SB_2_grid",
                    rz.O_THROOT: "MeanThetaInRootZone", rz.O_ROOT: "ROOTgrid", rz.O_S0: "gw_storage_coeff_1",
                    rz.O_ETR2ETP: "ETR2ETP"})
    # glacier cells: only the cleared fluxes are written by unsatzon
    cleared = {rz.O_EVAR, rz.O_INFEX, rz.O_INFILT, rz.O_QD, rz.O_QI, rz.O_POND, rz.O_PERCOL, rz.O_UPRISE,
               rz.O_GWN, rz.O_MACRO, rz.O_SUBSTEPS, rz.O_DELTAGW, rz.O_GWIN, rz.O_GWEX}

    nsteps = d(PRE, "theta").shape[0]
    last = nsteps if args.steps == 0 else min(nsteps, args.first + args.steps)
    comp = kind == rz.COLUMN
    glac = kind == rz.GLACIER
    worst = {}
    ndiff = {}
    t_run = []
    iters = []
    splits = []
    for t in range(args.first, last):
        inputs = dict(static)
        inputs.update(vbg=np.array([g(PRE, f"vegetation_coverage_degree{v}", t) for v in range(1, nv + 1)]),
                      wurz=np.array([g(PRE, f"root_depth{v}", t) for v in range(1, nv + 1)]),
                      etp=np.array([g(PRE, f"pot_transpiration{v}", t) for v in range(nv + 1)]),
                      prec=g(PRE, prec_grid, t), evap=g(PRE, "potential_evaporation", t),
                      snow=g(PRE, "snowstorage_solid", t) if opts.snow else np.zeros(n))
        inputs = {k: np.ascontiguousarray(v) for k, v in inputs.items()}
        out = np.zeros((rz.NOUT, n))
        for i, name in scalar_in.items():
            out[i] = g(PRE, name, t)
        state = dict(theta=stack(PRE, "theta", t), hh=stack(PRE, "hydraulic_heads", t), hgeo=hgeo.copy(),
                     qu=stack(PRE, "q_down", t),
                     etr=np.ascontiguousarray([g(PRE, f"real_transpiration{v}", t) for v in range(nv + 1)]), out=out)
        if rft is not None:
            rftab = rz.RiverFlowTable(*(np.ascontiguousarray(rft["rft_run_start"][t][:, j]) for j in range(4)))
        start = time.perf_counter()
        rz.richards(opts, tables, row_start, kind, inputs, state, python=args.python, order=order, lnk=lnk,
                    table=rftab)
        t_run.append(time.perf_counter() - start)
        iters.append(out[rz.O_ITER][comp].sum())
        splits.append(out[rz.O_SUBSTEPS][comp].sum())

        def check(name, mine, ref, sel):
            diff = np.abs(mine - ref)[sel]
            if diff.ndim > 1:
                diff = diff.max(axis=1)
            bad = ~((mine == ref) | (np.isnan(mine) & np.isnan(ref)))[sel]
            if bad.ndim > 1:
                bad = bad.any(axis=1)
            worst[name] = max(worst.get(name, 0.0), float(np.nanmax(diff)) if diff.size else 0.0)
            ndiff[name] = ndiff.get(name, 0) + int(bad.sum())
            if args.show and bad.any() and ndiff[name] - int(bad.sum()) < args.show:
                idx = np.flatnonzero(sel)[bad][:args.show]
                for c in idx:
                    print(f"  step {t} {name} cell {c} (row {cells.rows[c]}, col {cells.cols[c]}): "
                          f"mine {np.atleast_1d(mine[c])[:nl + 1]} WaSiM {np.atleast_1d(ref[c])[:nl + 1]}")

        for name, key in (("theta", "theta"), ("hydraulic_heads", "hh"), ("q_down", "qu")):
            check(name, state[key][:, :nl + 1], g(POST, name, t), comp)
        for v in range(nv + 1):
            check(f"real_transpiration{v}", state["etr"][v], g(POST, f"real_transpiration{v}", t), comp | glac)
        for i, name in compare.items():
            check(name, out[i], g(POST, name, t), comp | (glac if i in cleared else np.zeros(n, bool)))
        if rftab is not None:
            every = np.ones(nz, bool)
            check("riverflow infilt_amount (run)", rftab.infilt, rft["rft_run_end"][t][:, 2], every)
            check("riverflow infilt_fraction (run)", rftab.infilt_frac, rft["rft_run_end"][t][:, 3], every)
            rz.gather_river_flow_table(rftab)
            check("riverflow infilt_amount (post_run)", rftab.infilt, rft["rft_post_run"][t][:, 2], every)
            check("riverflow infilt_fraction (post_run)", rftab.infilt_frac, rft["rft_post_run"][t][:, 3], every)

    steps = last - args.first
    print(f"grid {mask.shape[0]} x {mask.shape[1]}, {n} cells: {comp.sum()} computed ({nlink} river-link cells "
          f"in their own row), {glac.sum()} fully glacierised, {(kind == rz.SKIP).sum()} not computed by WaSiM; "
          f"{nl} levels, {steps} steps, "
          f"{'plain Python' if args.python else 'DaCe'}, {'independent columns (fix)' if args.fix else 'row buffer carried'}")
    print(f"sub-steps per cell and step {np.sum(splits) / steps / max(comp.sum(), 1):.2f}, "
          f"iterations per cell and step {np.sum(iters) / steps / max(comp.sum(), 1):.2f}")
    warm = t_run[1:] if len(t_run) > 1 else t_run
    print(f"{1000 * np.mean(warm):.1f} ms per call (warm, without gather; median {1000 * np.median(warm):.1f}, "
          f"min {1000 * np.min(warm):.1f}), first call {t_run[0]:.1f} s")
    print("largest absolute difference to WaSiM over all steps and cells (number of differing cell-steps):")
    for name in worst:
        print(f"  {name:28s} {worst[name]:.3g}  ({ndiff[name]})")
    return 0 if all(v == 0 for v in ndiff.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
