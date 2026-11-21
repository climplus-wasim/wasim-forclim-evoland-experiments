#!/usr/bin/env python3
"""Make the WaSiM grid dumps that gt4py/chained.py compares the driver with.

One WaSiM run per dump point, each dumping only the grids compared or fed at that point, so that the
dumps of 48 Doubs steps fit in a few GB (all grids at all points would be about 30 GB). The runs are
deterministic (one thread), so the dumps of different runs belong to the same trajectory. The binary
must have the grid dump hook (wasim-eth eab8916) and, for the routing comparison, the routing dump
hook (routing/wasim-routing-dump.patch); the routing dump is written by the run of after_routing.

usage: chained_dumps.py --binary WASIM --template TEMPLATE --case-dir CASE --dump DUMP_DIR --runs RUN_DIR
                        [--case doubs|rhone] [--points p1,p2]
"""
import argparse
import os
import subprocess
import sys

LAYERS = (0, 1, 2)
VEG = [f"{g}{v}" for v in (1, 2) for g in ("landuse", "leaf_area_index", "vegetation_coverage_degree",
                                           "SurfaceCanopyResistance", "SurfaceIntercepResistance",
                                           "RoughnessLength", "root_depth")]
METEO = ["temperature", "precipitation", "wind_speed", "air_humidity", "global_radiation", "sunshine_duration"]
ETP = ([f"pot_transpiration{v}" for v in LAYERS] + [f"potential_InterceptionEvap{v}" for v in LAYERS]
       + ["potential_evaporation", "AspectCorrectedGlobalRadiation", "cloudiness", "global_radiation",
          "corr_temperature", "SSDTemp", "temperature_night"])
ICPT = ([f"interception_storage{v}" for v in LAYERS] + [f"interception_evaporation{v}" for v in LAYERS]
        + [f"throughfall{v}" for v in LAYERS] + [f"pot_transpiration{v}" for v in LAYERS])
SNOW = ["snowstorage_solid", "snowstorage_liquid", "snow_storage_total", "snowtemp", "surftemp", "snow_age",
        "days_snow", "albedo", "snow_fraction", "rain_fraction", "snowcover_outflow", "melt_from_snowcover",
        "snow_evaporation", "precipitation", "throughfall0"]
GLACIER = ["GlacierizedCells", "GlacierizedCellsOld", "GlacierCodes", "GlacierCodesOld", "GlacierMassBalance",
           "OldGlacierMassBalance", "firn_WE_stack", "SolidSnowStorage_onGlacier", "LiquidSnowStorage_onGlacier",
           "TotalSnowStorage_onGlacier", "snowtemp_onGlacier", "surftemp_onGlacier", "SnowEvaporation_onGlacier",
           "SnowMelt_onGlacier", "melt_from_ice", "melt_from_firn"]
SLIDES = ["input_mass", "mobile_mass", "deposition"]
SOIL = (["groundwater_distance", "groundwater_level", "Theta_in_GWLEVEL", "baseflow", "ponding_storage",
         "real_evaporation", "SB_1_grid", "SB_2_grid", "MeanThetaInRootZone", "ROOTgrid", "macropore_infiltration",
         "number_of_substeps", "groundwater_recharge", "Capillar_Uprise", "Percolation", "directflow", "interflow",
         "infiltration_excess", "total_infiltration", "river_infiltration", "river_exfiltration",
         "gw_storage_coeff_1", "ETR2ETP", "q_downext", "PrecOnLakes"]
        + [f"real_transpiration{v}" for v in LAYERS])
STACKS = ["theta", "hydraulic_heads", "q_down"]
GW = ["gw_head_1", "gw_head_old_1", "gw_head_old_old_1", "gw_transmissivity_x_1", "gw_transmissivity_y_1",
      "gw_flow_x_1", "gw_flow_y_1", "gw_balance1", "gw_leakage_1", "aquifer_thickness_1", "aquifer_upper_limit1",
      "gw_k_x_1", "gw_k_y_1", "gw_boundary_fix_h_1", "gw_boundary_fix_q_1", "CellSizeX", "CellSizeY"]
STATIC = ["[zone_grid]", "[elevation_model]", "slope_angle", "slope_aspect", "zonegrid_soilmodel", "soil_types",
          "lake_codes", "max_ponding_storage", "river_depth", "river_width", "river_links", "kolmation",
          "tangens_beta", "geodetic_heads", "deposition_index", "slidefraction1", "slidefraction2", "slidefraction3",
          "slidefraction4", "max_deposition", "elev_order_row", "elev_order_col", "landuse",
          "SurfaceEvaporationResistance", "debris_on_glaciers"]


def plan(case):
    glacier = GLACIER if case == "rhone" else []
    everything = sorted(set(METEO + VEG + ETP + ICPT + SNOW + glacier + SLIDES + SOIL + STACKS + GW + STATIC
                            + ["albedo", "Shadow", "radiation_correction", "Ieff_max"]))
    return {
        "start": everything,
        "after_meteo": METEO + VEG + ["albedo", "SurfaceEvaporationResistance", "SSDTemp"],
        "after_expokorr": ["radiation_correction", "Ieff_max", "corr_temperature", "Shadow", "temperature"],
        "after_etp": ETP,
        "after_interception": ICPT + ["precipitation", "snowcover_outflow"],
        "after_snow1": SNOW + glacier + ["ponding_storage"],
        "after_snowevap": [f"pot_transpiration{v}" for v in LAYERS] + ["potential_evaporation", "snowstorage_solid",
                                                                      "snow_evaporation"] + glacier,
        "after_unsatzon": SOIL + STACKS + ["snowstorage_solid", "snowcover_outflow"],
        "after_gwflow": SOIL + GW + glacier + ["snow_storage_total", "snowstorage_liquid", "snowstorage_solid"],
        "after_snow2": SNOW + SLIDES + glacier,
        "after_post_run": SOIL + SNOW + glacier + STACKS,
        "after_routing": SOIL + ["albedo"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--case", default="doubs", choices=("doubs", "rhone"))
    ap.add_argument("--points", default=None)
    ap.add_argument("--run-case", default="/home/user/wasim-eth/tests/golden/run_case.py")
    args = ap.parse_args()
    points = plan(args.case)
    wanted = args.points.split(",") if args.points else list(points)
    os.makedirs(args.dump, exist_ok=True)
    for point in wanted:
        out = os.path.join(args.runs, point)
        env = dict(os.environ, WASIM_DUMP_GRIDS=",".join(points[point]), WASIM_DUMP_DIR=args.dump,
                   WASIM_DUMP_POINTS=point)
        if point == "after_routing":
            env["WASIM_ROUTING_DUMP"] = os.path.join(args.dump, "routing")
            os.makedirs(env["WASIM_ROUTING_DUMP"], exist_ok=True)
        if point == "start":
            env["WASIM_TIMING"] = os.path.join(args.runs, "timing.tsv")
        print(point, flush=True)
        rc = subprocess.run([sys.executable, args.run_case, "--binary", args.binary, "--template", args.template,
                             "--case-dir", args.case_dir, "--out", out], env=env).returncode
        if rc:
            raise SystemExit(f"{point}: WaSiM failed ({rc})")


if __name__ == "__main__":
    main()
