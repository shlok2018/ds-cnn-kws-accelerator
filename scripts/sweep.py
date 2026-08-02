#!/usr/bin/env python3
# ============================================================================
# sweep.py
# The Phase 1 workhorse: sweep the accelerator design space, collect
# energy / latency / area / utilization for every (design point x layer),
# dump one tidy CSV you can load in the notebook.
#
# It uses the TimeloopFE Python interface (pytimeloop.timeloopfe.v4), which is
# the modern way to drive Timeloop from Python. The joblib.Parallel pattern is
# straight from the official docs -- the mapper search is the slow part, so
# parallelizing across design points is the difference between minutes and
# an afternoon.
#
# RUN THIS INSIDE THE DOCKER CONTAINER
#   (Accelergy-Project/accelergy-timeloop-infrastructure). Native installs of
#   Timeloop are a rite of passage nobody enjoys. Don't.
#
#   $ docker-compose run infrastructure       # or the arch-specific variant
#   # ... then inside:
#   $ cd /path/to/dscnn-accel
#   $ python3 scripts/make_problems.py        # generate the layer problems
#   $ python3 scripts/sweep.py                # run the sweep -> results.csv
#
# WHAT TO EXPECT: the pointwise and standard layers will look efficient;
# depthwise will show poor PE utilization that gets WORSE as the array grows.
# That inversion (16x16 worse than 8x8 on depthwise) is the headline finding.
# ============================================================================

import os
import re
import csv
import copy
import itertools

from joblib import Parallel, delayed

# TimeloopFE v4 interface. If this import fails, you're not in the container
# or the package name has drifted -- check `pip show pytimeloop`.
import pytimeloop.timeloopfe.v4 as tl

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.abspath(os.path.join(HERE, ".."))
TOP       = os.path.join(ROOT, "top.yaml.jinja2")   # assembles the full v4 spec
PROBLEMS  = os.path.join(ROOT, "problems")
OUT_CSV   = os.path.join(ROOT, "results.csv")

# The jinja template resolves its !include paths relative to the working dir,
# so pin CWD to the kit root. Runs at import time too, so joblib workers (which
# re-import this module) inherit it.
os.chdir(ROOT)

# ---- THE DESIGN SPACE ------------------------------------------------------
# Keep it small at first. This is 4 arrays x 3 buffer sizes x 3 layers = 36
# runs, which is plenty to start and won't melt your laptop. Widen once the
# pipeline works end-to-end.
ARRAY_SHAPES = [(4, 4), (8, 8), (8, 16), (16, 16)]   # (meshX, meshY)
GLB_DEPTHS   = [1024, 2048, 4096]                    # global buffer rows
LAYERS       = ["conv_standard", "conv_depthwise", "conv_pointwise", "conv_fc"]


def run_one(meshX, meshY, glb_depth, layer, constraints=None, dataflow="ws",
            arch=None, victory=None):
    """Evaluate a single (design point, layer) and return a result dict.

    constraints: optional path (relative to ROOT) to a dataflow-constraint file;
                 None uses the template default (weight_stationary.yaml).
    arch:        optional path to an arch file; None uses baseline_8x8.yaml.
                 Pass arch/baseline_8x8_cxm.yaml for the rigid C x M systolic pin.
    dataflow:    short tag distinguishing runs/CSV rows for different dataflows.
    """
    # Load the full spec fresh each time (TimeloopFE objects are mutable and
    # we're editing them, so no sharing across parallel workers). The template
    # pulls in globals / top-level variables / the compound-component library;
    # `problem` selects this layer's workload (path is relative to ROOT/CWD).
    rel_problem = os.path.join("problems", layer + ".yaml")
    jpd = {"problem": rel_problem}
    if constraints:
        jpd["constraints"] = constraints
    if arch:
        jpd["arch"] = arch
    spec = tl.Specification.from_yaml_files(TOP, jinja_parse_data=jpd)

    # Optional: search harder than the sweep default (mapper.yaml victory=500).
    # Use for the FINAL chosen design point to get a properly-optimized number.
    if victory is not None:
        spec.mapper.victory_condition = victory

    # --- patch the design point into the loaded spec ---
    # Walk to the PE container and set its spatial mesh.
    pe = spec.architecture.find("PE")
    pe.spatial.meshX = meshX
    pe.spatial.meshY = meshY

    # Set the global buffer depth.
    glb = spec.architecture.find("global_buffer")
    glb.attributes["depth"] = glb_depth

    # Unique output dir per point so parallel runs don't stomp each other.
    tag = f"{layer}_{meshX}x{meshY}_glb{glb_depth}_{dataflow}"
    outdir = os.path.join(ROOT, "runs", tag)
    os.makedirs(outdir, exist_ok=True)

    try:
        # This kicks off Accelergy (energy/area tables) + Timeloop mapper.
        stats = tl.call_mapper(spec, output_dir=outdir)
    except Exception as e:            # a bad design point shouldn't kill the sweep
        return dict(layer=layer, meshX=meshX, meshY=meshY, glb_depth=glb_depth,
                    dataflow=dataflow, ok=False, error=str(e)[:200])

    # call_mapper returns a pytimeloop OutputStats object. The attribute names
    # AND UNITS were verified against this image (timeloop-accelergy-pytorch);
    # the older field-name/unit guesses were wrong on three counts, so this is
    # pinned deliberately. If a future image changes them, open
    # runs/<tag>/timeloop-mapper.stats.txt and re-map here once.
    def g(obj, *names):
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return None

    #   computes            : int    (MACs)
    #   energy              : float, JOULES         -> uJ  = *1e6
    #   cycles              : int
    #   area                : float, m^2            -> um^2 = *1e12
    #   percent_utilization : float, already a %    (0..100)
    macs     = g(stats, "computes", "algorithmic_computes")
    energy_J = g(stats, "energy", "total_energy")
    cycles   = g(stats, "cycles", "total_cycles")
    area_m2  = g(stats, "area", "total_area")
    util     = g(stats, "percent_utilization", "utilization")

    energy_per_infer_uJ = (energy_J * 1e6) if energy_J is not None else None
    area_um2 = (area_m2 * 1e12) if area_m2 is not None else None

    # Operand footprints at DRAM (= the algorithmic tensor sizes) for the
    # roofline. OutputStats.mapping carries a line like:
    #   "DRAM [ Weights:4096 (4096) Inputs:8000 (8000) Outputs:8000 (8000) ]"
    footprint_bytes = arith_intensity = None
    mp = getattr(stats, "mapping", None)
    if mp:
        m = re.search(r"DRAM \[ Weights:(\d+).*?Inputs:(\d+).*?Outputs:(\d+)", mp)
        if m:
            footprint_bytes = sum(int(x) for x in m.groups())  # int8 -> 1 byte/elem
    if macs and footprint_bytes:
        arith_intensity = macs / footprint_bytes   # MACs per byte moved (algorithmic)

    return dict(
        layer=layer, meshX=meshX, meshY=meshY, glb_depth=glb_depth,
        dataflow=dataflow, ok=True,
        macs=macs,
        energy_uJ=energy_per_infer_uJ,
        cycles=cycles,
        area_um2=area_um2,
        pe_utilization=util,
        footprint_bytes=footprint_bytes,
        arith_intensity=arith_intensity,
        # energy-delay product: the single-number figure of merit architects love
        edp=(energy_per_infer_uJ * cycles) if (energy_per_infer_uJ and cycles) else None,
    )


def main():
    grid = list(itertools.product(ARRAY_SHAPES, GLB_DEPTHS, LAYERS))
    print(f"sweeping {len(grid)} design points "
          f"({len(ARRAY_SHAPES)} arrays x {len(GLB_DEPTHS)} buffers x {len(LAYERS)} layers)")

    # n_jobs=-1 uses all cores. Drop to a fixed number if your machine chokes.
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one)(mx, my, d, layer)
        for (mx, my), d, layer in grid
    )

    # Union of keys across all rows (failed rows have fewer).
    fields = []
    for r in results:
        for k in r:
            if k not in fields:
                fields.append(k)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"\ndone: {ok}/{len(results)} points succeeded -> {OUT_CSV}")
    if ok < len(results):
        print("some points failed; check the 'error' column and the runs/ dir.")


if __name__ == "__main__":
    main()
