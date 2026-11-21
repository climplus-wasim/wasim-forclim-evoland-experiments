#!/usr/bin/env python3
"""Chained validation of wasim-dace's driver (wasim_gt.driver, #14) against WaSiM grid dumps.

The driver starts from WaSiM's state at the start of the first step (dump point `start`, step 0) and
runs N steps with its own state carried from step to step: every process gets its inputs from the
driver's state and the hand-overs of the processes before it, never from the dumps. After each process
the driver's state and hand-overs are compared with WaSiM's dump at the same point, every step and
every model cell, exactly (NaN, infinity and nodata patterns included). The first difference is
reported with its step, point, grid and cell.

Processes that are not ported (the gaps) are fed from the dumps at the point where WaSiM computes them,
so that the comparison of everything else stays exact:
  - meteo grids: the meteo interpolation (ported and validated by meteo_interpolation.py) has no
    state of the model, so the driver takes its result as forcing; here the after_meteo grids
    (temperature, precipitation, wind, humidity, global radiation, sunshine duration);
  - the land-use albedo the snow model asks the land-use table for (LUSEtabletype::GetAlbedo, not
    ported), read back from the dumps as in snow_energy_balance.py;
  - river-link cells of surface interaction 2 (Doubs; #8, branch claude/8-river-links): the soil
    model's results on these cells are taken from after_unsatzon (they are not compared);
  - the zonal post-runs (travel-time zones, linear reservoirs, the runoff per subbasin) and the
    river-flow table's infiltration: the routing's inputs qges_soil, infilt and infilt_frac and the
    lake-cell inputs q_downext and PrecOnLakes come from the routing dump (routing_in); every grid the
    post-runs change is compared at after_post_run and then fed (none should be).

Dumps: chained_dumps.py (one WaSiM run per dump point, the binary with the grid and routing dump hooks):
  python3 chained_dumps.py --binary WASIM --template doubs48.wasim.in --case-dir Doubs --dump DUMP --runs RUNS
then
  NPY_DISABLE_CPU_FEATURES=X86_V4 uv run python chained.py DUMP RUNS [--steps 48] [--case doubs]
RUNS/<point>/control.wasim(.intern) are the expanded control files of the runs (all identical).

The driver's step calls the compiled programs through their Python wrappers; the time per step and per
process (without dump I/O and comparisons) is printed, with WaSiM's WASIM_TIMING of the same run if
RUNS/timing.tsv exists.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "routing"))

from wasim_gt.cells import CellLayout
from wasim_gt.driver import model as dm
from wasim_gt.driver import state as ds
from wasim_gt.processes import groundwater as gwp
from wasim_gt.processes import potential_et as pet
from wasim_gt.processes import routing as rp
from wasim_gt.processes import routing_tables as rt
from wasim_gt.processes import vegetation as veg
from wasim_gt.processes.radiation_correction import Shading
from wasim_gt.processes.snow_slides import SLIDE_NEIGHBOURS, visiting_order
from wasim_gt.processes.sun import Site
from wasim_gt.soil import richards as rz

import groundwater as gwv
import potential_et as petv
import richards as rzv
import vegetation as vegv
from dump import RoutingDump
from glacier_melt import debris_scaling, ice_firn_parameters
from snow_energy_balance import snow_parameters
from wasim_gt.processes.glacier_melt import GlacierMeltParameters
from validate import Mapping, compare_state, config_from_run, inflow_series, lake_inputs, state_from_dump
from wasim_dump import Dump, landuse_table

NODATA = -9999.0
LAYERS = 2

SOIL_ROWS = {rz.O_GWLEVEL: "groundwater_level", rz.O_GWSTAND: "groundwater_distance", rz.O_GWTHETA: "Theta_in_GWLEVEL",
             rz.O_DELTAGW: "baseflow", rz.O_POND: "ponding_storage", rz.O_EVAR: "real_evaporation",
             rz.O_QD: "directflow", rz.O_QI: "interflow", rz.O_GWEX: "river_exfiltration",
             rz.O_GWIN: "river_infiltration", rz.O_MACRO: "macropore_infiltration", rz.O_GWN: "groundwater_recharge",
             rz.O_PERCOL: "Percolation", rz.O_UPRISE: "Capillar_Uprise", rz.O_INFEX: "infiltration_excess",
             rz.O_INFILT: "total_infiltration", rz.O_SUBSTEPS: "number_of_substeps", rz.O_SB1: "SB_1_grid",
             rz.O_SB2: "SB_2_grid", rz.O_THROOT: "MeanThetaInRootZone", rz.O_ROOT: "ROOTgrid",
             rz.O_S0: "gw_storage_coeff_1", rz.O_ETR2ETP: "ETR2ETP"}
STACKS = {"theta": "theta", "hydraulic_heads": "hh", "q_down": "qu"}
SNOW = {"snowstorage_solid": "solid", "snowstorage_liquid": "liquid", "snow_storage_total": "total",
        "snowtemp": "temperature", "surftemp": "surface_temperature", "snow_age": "age", "days_snow": "days"}
GLACIER = {"GlacierizedCells": "ice", "GlacierMassBalance": "mass_balance", "SolidSnowStorage_onGlacier": "solid",
           "LiquidSnowStorage_onGlacier": "liquid", "TotalSnowStorage_onGlacier": "total",
           "snowtemp_onGlacier": "temperature", "surftemp_onGlacier": "surface_temperature"}
VEG = {"leaf_area_index": "lai", "vegetation_coverage_degree": "vcf", "SurfaceCanopyResistance": "rsc",
       "SurfaceIntercepResistance": "rsi", "RoughnessLength": "z0", "root_depth": "root_depth"}
GW = {"gw_head_1": "head", "gw_head_old_1": "old_head", "gw_flow_x_1": "flow_x", "gw_flow_y_1": "flow_y",
      "gw_balance1": "balance"}


def text_of(path):
    return open(path, encoding="latin-1").read()


def section(text, name):
    body = text.split(f"[{name}]", 1)[1].split("\n[", 1)[0]
    return [ln for ln in (x.split("#", 1)[0].strip() for x in body.splitlines()) if ln]


def landuse_albedo(d, active, first_hour, nsteps, steps_per_day=24):
    """The land-use albedo the snow model gets, per step (snow_energy_balance.landuse_albedo without the
    count of unobserved cells): read back from after_snow1 where a cell ended the step snow-free."""
    meteo, snow1, swe1 = d("after_meteo", "albedo"), d("after_snow1", "albedo"), d("after_snow1", "snow_storage_total")
    day = (np.arange(nsteps) + first_hour) // steps_per_day
    out = [None] * nsteps
    for k in np.unique(day):
        steps = np.flatnonzero(day == k)
        value = meteo[steps[0]].copy()
        for t in steps:
            free = active & (swe1[t] <= 5.0)
            value[free] = snow1[t][free]
        for t in steps:
            out[t] = value
    return out


class Setup:
    """Static configuration and initial state of the driver from the dumps and the control files."""

    def __init__(self, dump_dir, runs, case):
        self.d = d = Dump(dump_dir)
        self.case = case
        self.control = os.path.join(runs, "start", "control.wasim")
        self.intern = self.control + ".intern"
        text = text_of(self.intern)
        active = d("start", "[zone_grid]")[0] > NODATA
        self.active = active
        self.cells = cells = CellLayout.from_mask(active)
        self.n = n = cells.n
        s0 = lambda name, dtype=np.float64: cells.to_cells(d("start", name)[0], dtype)  # noqa: E731
        self.s0 = s0
        self.nsteps = d("after_routing", "ponding_storage").shape[0]

        # time and site (potential_et.py)
        coords, out_iv = section(text, "coordinates"), section(text, "output_interval")
        et = section(text, "evapotranspiration")
        self.step_minutes = step = float(et[1])
        hour_offset = float(out_iv[3].split()[0]) / 60.0
        site = Site(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), hour_offset)
        mt = section(text, "model_time")
        hour, day, month, year = (int(x) for x in mt[:4])
        self.start = dt.datetime(year, month, day) + dt.timedelta(hours=hour)
        self.first_hour = hour

        # radiation correction and potential ET
        rc, lake = section(text, "radiation_correction"), section(text, "lake_model")
        lakes_on = int(lake[0]) != 0
        amp = petv.numbers(et[30])
        et_params = pet.Parameters(cr=tuple(petv.numbers(et[25])), dt_koeff=petv.numbers(et[28])[0],
                                   dtmax=tuple(petv.numbers(et[29])[:12]), ampl_day=amp[0], ampl_annual=amp[1],
                                   albedo_open_water=float(lake[2]) if lakes_on else 0.0,
                                   z0_open_water=float(lake[3]) if lakes_on else 0.0, lakes=lakes_on)
        dem_grid = d("start", "[elevation_model]")[0]
        m = re.search(r"cellsize\s+([0-9.]+)", open(section(text, "elevation_model")[0], "rb").read(400).decode("latin-1"))
        cellsize = float(m.group(1)) if m else 500.0
        radiation = dm.RadiationConfig(int(rc[2]), float(rc[5]), int(rc[11]), Shading.from_grid(cells, dem_grid),
                                       s0("slope_angle"), s0("slope_aspect"), cellsize)
        luse0 = d("start", "landuse")[0]
        kl, ls = np.full(luse0.shape, 0.3), np.full(luse0.shape, 10.0)
        for code, (k, sc) in petv.multilayer_table(text).items():
            kl[luse0 == code], ls[luse0 == code] = k, sc
        dem = s0("[elevation_model]")
        lake_codes = s0("lake_codes")
        et_cfg = dm.EtConfig(et_params, cells.to_cells(kl), cells.to_cells(ls), dem, lake_codes)

        # vegetation (vegetation.py) and interception capacity (interception.py)
        table = vegv.landuse_table(text)
        veg_cfg = dm.VegetationConfig(table, veg.Table.from_table(table), dem, hour_offset)
        landuse = np.array([s0(f"landuse{v}") for v in range(1, LAYERS + 1)])
        cap_table = landuse_table(self.control, "IntercepCap")
        capacity = np.full(landuse.shape, 0.35)
        for code, value in cap_table.items():
            if value >= 0:
                capacity[landuse == code] = value
        icpt = int(section(text, "interception_model")[2].split()[0])

        # snow (snow_energy_balance.py)
        snow_params, slide = snow_parameters(self.control)
        slides = dict(zone=s0("[zone_grid]"), slope=s0("slope_angle"),
                      slide_fractions=[s0(f"slidefraction{k}") for k in range(1, 5)], deposition_max=s0("max_deposition"),
                      order=visiting_order(cells, d("start", "elev_order_row")[0], d("start", "elev_order_col")[0]),
                      neighbours=cells.neighbours(SLIDE_NEIGHBOURS), **slide)
        snow_cfg = dm.SnowConfig(snow_params, s0("deposition_index"), dem, lakes_on, slides)
        self.lu_albedo = landuse_albedo(d, active, self.first_hour, self.nsteps)

        # soil (richards.py)
        opts, fd_zone, sr_zone = rzv.unsatzon_options(text)
        opts.cellsize = cellsize
        tables = rz.soil_tables(rzv.soil_types(text))
        self.levels = nl = tables.levels
        lu = rzv.landuse_parameters(text)
        kind = np.full(n, rz.COLUMN, dtype=np.int64)
        self.has_glacier = d.has("start", "GlacierizedCells")
        if self.has_glacier:
            kind[np.clip(s0("GlacierizedCells"), 0.0, 1.0) == 1.0] = rz.GLACIER
        if opts.surface_interaction > 1:
            kind[s0("river_links") != NODATA] = rz.SKIP
        self.kind = kind
        zcode = s0("zonegrid_soilmodel").astype(np.int64)
        zgrid = d("start", "zonegrid_soilmodel")[0]
        for tbl in (fd_zone, sr_zone):
            prev = None
            for z in np.unique(zgrid[zgrid > NODATA + 1]).astype(int):
                prev = tbl.get(z, prev)
                tbl[z] = prev
        lu_par = lambda j: np.array([[lu[int(c)][j] if c > NODATA else 0.0 for c in x] for x in landuse])  # noqa: E731
        static = dict(soil=s0("soil_types").astype(np.int64), zone_fd=np.array([fd_zone[z] for z in zcode]),
                      zone_sr=np.array([sr_zone[z] for z in zcode]),
                      lu_valid=np.array([(x > NODATA).astype(np.int64) for x in landuse]),
                      lu_formpar=lu_par(0), lu_redudry=lu_par(1), lu_reduwet=lu_par(2), lu_limitwet=lu_par(3),
                      lake=lake_codes, maxpond=s0("max_ponding_storage"), rivdep=s0("river_depth"),
                      rivwit=s0("river_width"), kolm=s0("kolmation"), tanbeta=s0("tangens_beta"))
        opts.snow = 1
        soil_cfg = dm.SoilConfig(opts, tables, rz.rows_of(cells.rows, opts.fix_htemp1), kind, static, icpt > 10)

        # groundwater (groundwater.py)
        gw_cfg = gwv.read_parameters(self.control)
        gw_params = gwp.Parameters(*(s0(k) for k in ("[zone_grid]", "[elevation_model]", "aquifer_thickness_1",
                                                     "aquifer_upper_limit1", "gw_k_x_1", "gw_k_y_1",
                                                     "gw_storage_coeff_1", "gw_boundary_fix_h_1",
                                                     "gw_boundary_fix_q_1", "CellSizeX", "CellSizeY")), **gw_cfg)

        # routing (routing/validate.py)
        rdump = os.path.join(dump_dir, "routing")
        self.din = RoutingDump(rdump, "routing_in")
        self.dout = RoutingDump(rdump, "routing_out")
        setup = RoutingDump(rdump, "routing_setup")
        rcfg, layout, lines = config_from_run(os.path.join(runs, "after_routing"), setup)
        if not (np.array_equal(layout.rows, cells.rows) and np.array_equal(layout.cols, cells.cols)):
            raise SystemExit("routing layout differs from the model cells")
        self.rtables = rt.build(rcfg)
        self.mapping = Mapping(self.rtables)
        self.zl = {c: row for c, row in enumerate(self.rtables.meta["ch_rows"])
                   if row["kind"] == rt.INFLOW and row["zlart"] == rt.ZL_EXTERN}
        self.zl_series = inflow_series(self.rtables, lines, self.zl)

        glacier_cfg = None
        if self.has_glacier:  # glacier_melt.py
            glac = ice_firn_parameters(self.control)
            debris, scaling = np.ones(n), 1.0
            if d.has("start", "debris_on_glaciers"):
                grid = s0("debris_on_glaciers")
                debris = np.where(grid > NODATA + 1, grid, 1.0)
                scaling = debris_scaling(self.control)
            glacier_cfg = dm.GlacierConfig(GlacierMeltParameters(t0=snow_params.t0, debris_scaling=scaling),
                                           glac["mf"], glac["ice_min"], glac["ice_max"], glac["snow_min"],
                                           glac["snow_max"], debris)
        self.config = dm.Config(step, site, radiation, et_cfg, veg_cfg, icpt, capacity, snow_cfg, soil_cfg, gw_params,
                                gwp.topology(cells), glacier_cfg, self.rtables)

    def initial_state(self):
        d, s0, n, nl = self.d, self.s0, self.n, self.levels
        stack = lambda name: np.pad(s0(name), ((0, 0), (0, 1)))  # noqa: E731  [N, L+1] -> [N, L+2]
        out = np.zeros((rz.NOUT, n))
        for row, name in SOIL_ROWS.items():  # states, and the fluxes of the last step (kept on glacier cells)
            out[row] = s0(name)
        layers = range(1, LAYERS + 1)
        vegetation = ds.Vegetation(landuse=np.array([s0(f"landuse{v}") for v in layers]),
                                   **{f: np.array([s0(f"{g}{v}") for v in layers]) for g, f in VEG.items()},
                                   rse=s0("SurfaceEvaporationResistance"))
        head = s0("gw_head_1")
        gw = gwp.State(head, s0("gw_head_old_1"), head.copy(), head.copy(), s0("gw_flow_x_1"), s0("gw_flow_y_1"),
                       s0("gw_balance1"), s0("baseflow"))
        state = ds.State(
            vegetation=vegetation, albedo=s0("albedo"), et=ds.Evapotranspiration(s0("SSDTemp")),
            interception=np.array([s0(f"interception_storage{v}") for v in range(LAYERS + 1)]),
            snow=ds.Snow(**{f: s0(g) for g, f in SNOW.items()}),
            soil=ds.Soil(stack("theta"), stack("hydraulic_heads"), stack("geodetic_heads"), stack("q_down"),
                         np.array([s0(f"real_transpiration{v}") for v in range(LAYERS + 1)]), out),
            groundwater=ds.Groundwater(gw, s0("gw_leakage_1")),
            routing=state_from_dump(self.rtables, self.din, 0, self.mapping))
        if self.has_glacier:
            state.glacier = ds.Glacier(firn=s0("firn_WE_stack"), **{f: s0(g) for g, f in GLACIER.items()})
        return state

    def forcing(self, t):
        d, c = self.d, self.cells.to_cells
        now = self.start + dt.timedelta(minutes=self.step_minutes * t)
        g = lambda name: c(d("after_meteo", name)[t])  # noqa: E731
        date = self.din.get("date", t).astype(int)
        routing = dict(qges_soil=self.din.get("qges_soil", t))
        st_in = state_from_dump(self.rtables, self.din, t, self.mapping)
        routing.update(infilt=st_in.infilt, infilt_frac=st_in.infilt_frac)
        zl = np.zeros(self.rtables.sizes["NC"])
        for ch in self.zl:
            zl[ch] = self.zl_series[ch].get(tuple(date[:4]), np.nan)
        routing["zl_value"] = zl
        lakes = lake_inputs(self.rtables, self.din, t, self.mapping, self.n)
        routing.update(qu=lakes[3], prec=lakes[5], expected_lake_inputs=lakes, date=date)
        return dm.Forcing(dm.Clock.of(now), g("temperature"), g("precipitation"), g("wind_speed"),
                          g("air_humidity"), g("global_radiation"), g("sunshine_duration"), c(self.lu_albedo[t]),
                          routing)


def mismatch(mine, ref):
    """Mask of cells that differ (NaN equals NaN), and the largest finite difference."""
    mine, ref = np.asarray(mine, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    bad = ~((mine == ref) | (np.isnan(mine) & np.isnan(ref)))
    if bad.ndim > 1:
        bad = bad.reshape(bad.shape[0], -1).any(axis=1)
    both = np.isfinite(mine) & np.isfinite(ref)
    diff = np.where(both, np.abs(mine - ref), np.where(mine == ref, 0.0, np.inf))
    return bad, float(diff.max()) if diff.size else 0.0


class Checker:
    """The hook: compares the driver with the dumps at each point and feeds the gaps."""

    def __init__(self, setup: Setup, show=3):
        self.s = setup
        self.t = 0
        self.results = {}      # (point, grid) -> [max diff, differing cell-steps, steps compared]
        self.first = None
        self.show = show
        self.ported = setup.kind != rz.SKIP
        self.everywhere = np.ones(setup.n, dtype=bool)
        self.compare_seconds = 0.0

    def dump(self, point, name):
        return self.s.cells.to_cells(self.s.d(point, name)[self.t])

    def check(self, point, name, mine, cells=None):
        if not self.s.d.has(point, name):
            return
        ref = self.dump(point, name)
        if ref.ndim > 1 and np.ndim(mine) > 1 and ref.shape[1] < np.shape(mine)[1]:
            mine = np.asarray(mine)[:, :ref.shape[1]]
        sel = self.everywhere if cells is None else cells
        bad, diff = mismatch(np.asarray(mine)[sel], ref[sel])
        r = self.results.setdefault((point, name), [0.0, 0, 0])
        r[0] = max(r[0], diff if bad.any() else 0.0)
        r[1] += int(bad.sum())
        r[2] += 1
        if bad.any():
            c = np.flatnonzero(sel)[np.flatnonzero(bad)[0]]
            info = dict(step=self.t, point=point, grid=name, cell=int(c), row=int(self.s.cells.rows[c]),
                        col=int(self.s.cells.cols[c]), mine=np.atleast_1d(np.asarray(mine)[c])[:4].tolist(),
                        wasim=np.atleast_1d(ref[c])[:4].tolist(), cells=int(bad.sum()))
            if self.first is None:
                self.first = info
            if r[1] == int(bad.sum()) and self.show:
                print(f"  first difference of {point}/{name}: {info}")

    def __call__(self, point, state, h):
        start = time.perf_counter()
        getattr(self, point)(state, h)
        self.compare_seconds += time.perf_counter() - start

    def after_meteo(self, state, h):
        v = state.vegetation
        for g, f in VEG.items():
            for k in range(LAYERS):
                self.check("after_meteo", f"{g}{k + 1}", getattr(v, f)[k])
        self.check("after_meteo", "SurfaceEvaporationResistance", v.rse)
        self.check("after_meteo", "albedo", state.albedo)

    def after_expokorr(self, state, h):
        for g in ("radiation_correction", "Ieff_max", "corr_temperature", "Shadow"):
            self.check("after_expokorr", g, h[g])

    def after_etp(self, state, h):
        for v in range(LAYERS + 1):
            self.check("after_etp", f"pot_transpiration{v}", h["pot_transpiration"][v])
            self.check("after_etp", f"potential_InterceptionEvap{v}", h["potential_InterceptionEvap"][v])
        for g in ("potential_evaporation", "AspectCorrectedGlobalRadiation", "cloudiness", "global_radiation",
                  "corr_temperature"):
            self.check("after_etp", g, h[g])
        self.check("after_etp", "SSDTemp", state.et.ssd_mean)

    def after_interception(self, state, h):
        for v in range(LAYERS + 1):
            self.check("after_interception", f"interception_storage{v}", state.interception[v])
            self.check("after_interception", f"interception_evaporation{v}", h["interception_evaporation"][v])
            self.check("after_interception", f"throughfall{v}", h["throughfall"][v])
            self.check("after_interception", f"pot_transpiration{v}", h["pot_transpiration"][v])

    def after_snow1(self, state, h):
        p = "after_snow1"
        for g, f in SNOW.items():
            self.check(p, g, getattr(state.snow, f))
        self.check(p, "albedo", state.albedo)
        for g in ("snow_fraction", "rain_fraction", "snowcover_outflow", "melt_from_snowcover", "snow_evaporation"):
            self.check(p, g, h[g])
        if "throughfall0" in h:
            self.check(p, "throughfall0", h["throughfall0"])
        if "precipitation" in h:
            self.check(p, "precipitation", h["precipitation"])
        self.glacier(p, state, h)

    def glacier(self, p, state, h, fluxes=True):
        g = state.glacier
        if g is None:
            return
        on = g.ice > 0
        for name, f in GLACIER.items():
            self.check(p, name, getattr(g, f), on)
        self.check(p, "firn_WE_stack", g.firn, on)
        if fluxes:
            for name in ("SnowEvaporation_onGlacier", "SnowMelt_onGlacier", "melt_from_ice", "melt_from_firn"):
                self.check(p, name, h[name], on)

    def after_snowevap(self, state, h):
        for v in range(LAYERS + 1):
            self.check("after_snowevap", f"pot_transpiration{v}", h["pot_transpiration"][v])
        self.check("after_snowevap", "potential_evaporation", h["potential_evaporation"])
        self.check("after_snowevap", "snowstorage_solid", state.snow.solid)
        self.glacier("after_snowevap", state, h)

    def soil(self, point, state, cells, feed):
        """Compare the soil on `cells`; with feed, take the dump's values elsewhere (river-link cells)."""
        soil = state.soil
        other = ~cells
        for row, g in SOIL_ROWS.items():
            self.check(point, g, soil.out[row], cells)
            if feed and other.any() and self.s.d.has(point, g):
                soil.out[row][other] = self.dump(point, g)[other]
        for g, f in STACKS.items():
            self.check(point, g, getattr(soil, f), cells)
            if feed and other.any() and self.s.d.has(point, g):
                ref = self.dump(point, g)
                getattr(soil, f)[other, :ref.shape[1]] = ref[other]
        for v in range(LAYERS + 1):
            self.check(point, f"real_transpiration{v}", soil.etr[v], cells)
            if feed and other.any():
                soil.etr[v][other] = self.dump(point, f"real_transpiration{v}")[other]

    def after_unsatzon(self, state, h):
        self.soil("after_unsatzon", state, self.ported, feed=True)

    def after_gwflow(self, state, h):
        p = "after_gwflow"
        gw = state.groundwater.state
        for g, f in GW.items():
            self.check(p, g, getattr(gw, f))
        self.check(p, "gw_head_old_old_1", gw.old_head)
        self.check(p, "gw_transmissivity_x_1", h["gw_transmissivity_x_1"])
        self.check(p, "gw_transmissivity_y_1", h["gw_transmissivity_y_1"])
        self.check(p, "gw_leakage_1", state.groundwater.leakage)
        self.soil(p, state, self.everywhere, feed=False)
        self.glacier(p, state, h, fluxes=False)

    def after_snow2(self, state, h):
        p = "after_snow2"
        for g in ("snowstorage_solid", "snowstorage_liquid", "snow_storage_total"):
            self.check(p, g, getattr(state.snow, SNOW[g]))
        for g in ("input_mass", "mobile_mass", "deposition"):
            self.check(p, g, h[g])
        self.glacier(p, state, h, fluxes=False)

    def after_post_run(self, state, h):
        """The post-runs are not ported: compare what they may change, then take WaSiM's values."""
        p = "after_post_run"
        self.soil(p, state, self.everywhere, feed=False)
        for g, f in SNOW.items():
            self.check(p, g, getattr(state.snow, f))
        self.glacier(p, state, h, fluxes=False)
        # what the routing will read on lake cells, against WaSiM's routing input
        lakes = self.s.forcing_routing["expected_lake_inputs"]
        lake = self.s.config.et.lake > 0
        mine = np.where(lake, np.array([state.soil.pond, state.soil.etr[0], h["interception_evaporation"][0],
                                        lakes[3], state.soil.out[rz.O_DELTAGW], lakes[5]]), 0.0)
        for k, g in enumerate(("POND", "ETR", "EI", "q_downext", "baseflow", "PrecOnLakes")):
            bad, diff = mismatch(mine[k][lake], lakes[k][lake])
            r = self.results.setdefault(("routing_in (lake cells)", g), [0.0, 0, 0])
            r[0], r[1], r[2] = max(r[0], diff if bad.any() else 0.0), r[1] + int(bad.sum()), r[2] + 1
            if bad.any() and self.first is None:
                self.first = dict(step=self.t, point="routing_in", grid=g, cells=int(bad.sum()))

    def after_routing(self, state, h):
        p = "after_routing"
        self.soil(p, state, self.everywhere, feed=False)
        self.check(p, "albedo", state.albedo)
        diffs = compare_state(self.s.rtables, state.routing, self.s.dout, self.t, self.s.mapping, state.soil.pond)
        for k, v in diffs.items():
            r = self.results.setdefault(("routing_out", k), [0.0, 0, 0])
            r[0], r[1], r[2] = max(r[0], v), r[1] + int(v != 0), r[2] + 1
            if v != 0 and self.first is None:
                self.first = dict(step=self.t, point="routing_out", grid=k, diff=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("runs", help="directory of the WaSiM runs of chained_dumps.py (control files, timing.tsv)")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--case", default="doubs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-check", action="store_true", help="time the driver only")
    args = ap.parse_args()
    if "NPY_DISABLE_CPU_FEATURES" not in os.environ:
        print("warning: NPY_DISABLE_CPU_FEATURES not set")

    t0 = time.perf_counter()
    setup = Setup(args.dump_dir, args.runs, args.case)
    state = setup.initial_state()
    model = dm.Model(setup.config, setup.n)
    checker = Checker(setup)
    nsteps = min(args.steps or setup.nsteps, setup.nsteps)
    print(f"{args.case}: {setup.n} cells ({int((setup.kind == rz.SKIP).sum())} river-link cells fed), "
          f"{setup.levels} soil levels, {nsteps} steps of {setup.step_minutes:g} min from {setup.start}; "
          f"setup {time.perf_counter() - t0:.1f} s")
    step_times = []
    for t in range(nsteps):
        checker.t = t
        forcing = setup.forcing(t)
        setup.forcing_routing = forcing.routing
        if tuple(forcing.routing["date"][:4]) != (forcing.clock.year, forcing.clock.month, forcing.clock.day,
                                                  forcing.clock.hour):
            print(f"  step {t}: routing date {forcing.routing['date'][:4]} != clock {forcing.clock}")
        checker.compare_seconds = 0.0
        start = time.perf_counter()
        model.step(state, forcing, None if args.no_check else checker)
        step_times.append(time.perf_counter() - start - checker.compare_seconds)
    warm = np.array(step_times[2:]) * 1000 if nsteps > 3 else np.array(step_times) * 1000
    print(f"driver step: {np.median(warm):.2f} ms median (warm, steps 3..; without dump I/O and comparison), "
          f"first step {step_times[0]:.1f} s")
    stage = {k: 1000 * float(np.median(v[2:] if len(v) > 3 else v)) for k, v in model.timers.items()}
    print("  per process (median ms per call, steps 3..): " + ", ".join(f"{k} {v:.2f}" for k, v in stage.items()))
    bad = {k: v for k, v in checker.results.items() if v[1]}
    print(f"compared {len(checker.results)} (point, grid) pairs; {len(bad)} with differences")
    for (point, grid), (diff, cells, steps) in sorted(checker.results.items()):
        if cells or args.json is None:
            print(f"  {point:24s} {grid:34s} max {diff:.3g}  differing cell-steps {cells}  steps {steps}")
    print("first divergence:", checker.first)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(steps=nsteps, cells=setup.n, step_ms=float(np.median(warm)), stages=stage,
                           first=checker.first,
                           results={f"{p}/{g}": v for (p, g), v in checker.results.items()}), f, indent=1)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
