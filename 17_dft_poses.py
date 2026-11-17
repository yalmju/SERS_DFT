"""
DFT adsorption energy of every adsorption pose (upright, flat, dissociated) at every surface density

04, 12 and 16 compute one pose per molecule and cell (the lowest-MLIP one). This stage adds all other
candidate poses so that pose and density effects can be compared on the same footing:

    E_ads,aq(X, pose) = E(Au+X) − E(Au) − [E_gas(X) + G_solv(X)]

Each pose structure is placed in the cell of an existing, completed bare-slab calculation and uses that
slab, its cutoff and k-points, and the gas references at the same cutoff:
  3x2, 3x3, 5x5   bare slab of 12 (results/dft_coverage)
  4x4             bare slab of 04 (results/dft_check)
  4x4 (ternary)   bare slab of 16 (results/dft_ternary118), for the single-molecule 16 structures
Poses already computed by 04, 12 or 16 are found by fingerprint and reused, so only new poses run pw.x.
A pose that leaves less than MIN_VACUUM Å of vacuum in the reference cell is not computed (reason recorded).
Poses that do not fit a cell (compete.csv) are recorded as not_available.
Option: DFT_POSE_CELLS=3x3,4x4 (subset of 3x2, 3x3, 4x4, 5x5).

Input:  data/results_compete/, data/results_fix/, data/results_ternary/, completed 04, 12 and 16 results
Output: results/dft_poses/ (QE files, *.meta.json, record.json, poses_log.txt); table in results/tables/poses.csv
"""
import csv, json, os
import numpy as np
import qe_tools as Q
from ase.io import read

CELLS = ["3x2", "3x3", "4x4", "5x5"]
SEL = Q.env_list("DFT_POSE_CELLS", CELLS)
NAMES = ["THI", "TBZ", "DQ"]
MIN_VACUUM = 9.0
REFS = {  # reference group -> (stage folder, record key of its bare slab)
    "3x2": ("dft_coverage", "3x2/Au"), "3x3": ("dft_coverage", "3x3/Au"), "5x5": ("dft_coverage", "5x5/Au"),
    "4x4": ("dft_check", "Au"), "4x4-ternary": ("dft_ternary118", "Au"),
}

ST = Q.Stage("poses", "dft_poses", "poses_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun")
table = Q.load_sssp(log)
MACE = Q.load_mace(log)
S04 = Q.structures_from_04()
ST.constants["G_solv_eV"] = Q.read_gsolv()
ST.constants["min_vacuum_A"] = MIN_VACUUM


def reference(group):
    """Completed bare-slab calculation of an earlier stage: (meta, structure), read from results/ only."""
    folder, key = REFS[group]
    path = os.path.join("results", folder, "record.json")
    if not os.path.exists(path): return None, f"{path} not found"
    it = next((x for x in json.load(open(path))["items"] if x["key"] == key), None)
    if not it or not it.get("calc_name"): return None, f"{folder}: {key} not in record"
    m = Q.read_meta(it["calc_dir"], it["calc_name"])
    if not m or m.get("status") != "completed" or \
            not Q.check_output(os.path.join(it["calc_dir"], f"{it['calc_name']}.out"))["complete"]:
        return None, f"{folder}: {key} not completed"
    return (m, read(os.path.join(it["calc_dir"], f"{it['calc_name']}.in"), format="espresso-in")), ""


def poses(cl):
    """(molecule, pose, source, file) candidates of one cell, plus compete.csv 'does not fit' entries."""
    out, nofit = [], []
    for n in NAMES:
        for p in ("up", "flat"):
            f = f"data/results_compete/{n}_{p}_{cl}.traj"
            if os.path.exists(f): out.append((n, p, "compete", f, cl))
    f = f"data/results_fix/THI_DMDTCx2_{cl}.traj"
    if os.path.exists(f): out.append(("THI", "diss", "fix", f, cl))
    if cl == "4x4" and os.path.exists("data/results_ternary/ternary.csv"):
        for r in csv.reader(open("data/results_ternary/ternary.csv")):
            if len(r) >= 7 and r[0] == "4x4" and r[2] in NAMES and r[4]:
                f = f"data/results_ternary/4x4_{r[2]}.traj"
                if os.path.exists(f): out.append((r[2], r[6], "ternary", f, "4x4-ternary"))
    have = {(n, p) for n, p, *_ in out}
    if os.path.exists("data/results_compete/compete.csv"):
        for r in csv.DictReader(open("data/results_compete/compete.csv")):
            if r["cell"] == cl and r["molecule"] in NAMES and r["converged"] != "True" and (r["molecule"], r["pose"]) not in have:
                nofit.append((r["molecule"], r["pose"], r.get("note") or "not converged"))
    return out, nofit


def place(at, ref):
    """Pose structure in the reference cell: same bottom-layer height and cell; returns (atoms, vacuum, error)."""
    a = at.copy(); a.set_constraint()
    au = [i for i, x in enumerate(a) if x.symbol == "Au"]
    rau = [i for i, x in enumerate(ref) if x.symbol == "Au"]
    if len(au) != len(rau): return None, None, f"{len(au)} Au atoms, reference slab has {len(rau)}"
    if np.abs(a.cell.array[:2, :2] - ref.cell.array[:2, :2]).max() > 1e-3: return None, None, "lateral cell differs from reference"
    a.positions[:, 2] += ref.positions[rau, 2].min() - a.positions[au, 2].min()
    a.set_cell(ref.cell.array.copy()); a.pbc = True
    vac = float(ref.cell[2, 2] - np.ptp(a.positions[:, 2]))
    return a, vac, ""


gas = {}
for cl in SEL:
    cand, nofit = poses(cl)
    log(f"\n=== {cl}: {len(cand)} pose structure(s) ===")
    for n, p, why in nofit:
        ST.add(f"{cl}/{n}@Au/{p}", None, "not_available", f"does not fit this cell ({why}, compete.csv)",
               cell=cl, molecule=n, pose=p, role="pose")
    for n, p, srcname, f, grp in cand:
        key = f"{cl}/{n}@Au/{p}/{srcname}"
        ref, why = reference(grp)
        if ref is None:
            log(f"  ✗ {key:<30s} reference slab: {why}")
            ST.add(key, None, "not_available", f"reference slab: {why}", cell=cl, molecule=n, pose=p, role="pose"); continue
        rmeta, rat = ref
        fp = rmeta["fingerprint"]; W0, R0, kpts = fp["ecutwfc"], fp["ecutrho"], tuple(fp["kpts"])
        rkey = f"ref/{grp}"
        if rkey not in ST.constants.setdefault("refs", {}):
            ST.constants["refs"][rkey] = f"{rmeta['workdir']}/{rmeta['name']}"
            area = float(np.linalg.norm(np.cross(rat.cell[0], rat.cell[1])))
            ST.add(rkey, rmeta, role="slab", group=grp, area_A2=area, mlip_eV=Q.mlip_energy(rat, MACE),
                   ecut=[W0, R0], kpts=list(kpts))
            log(f"  reference {grp}: {rmeta['workdir']}/{rmeta['name']} (ecut {W0:g}/{R0:g} Ry, k {kpts})")
        gkey = f"gas{W0:g}/{n}"
        if gkey not in gas:
            g = Q.gas_reference(n, table, (W0, R0), f"{ST.dir}/gas_ecut{W0:g}", log)
            gas[gkey] = g
            ST.add(gkey, g, system=f"{n}_gas", molecule=n, role="gas",
                   mlip_eV=Q.mlip_energy(S04[f"{n}_gas"], MACE, periodic=False) if f"{n}_gas" in S04 else None)
        at, vac, err = place(read(f), rat)
        info = dict(cell=cl, molecule=n, pose=p, role="pose", ref=rkey, gas=gkey, group=grp, structure=os.path.basename(f))
        if err or vac < MIN_VACUUM:
            why = err or f"vacuum {vac:.1f} Å in the reference cell (< {MIN_VACUUM:g} Å)"
            log(f"  ✗ {key:<30s} {why}"); ST.add(key, None, "not_available", why, **info); continue
        src = Q.source(f, pose=p, origin=srcname, reference_slab=ST.constants["refs"][rkey], vacuum_A=round(vac, 2))
        pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
        name = f"{n}_{p}_{grp.replace('-', '_')}"
        meta = Q.run_or_reuse(f"{ST.dir}/{grp}", name, at, pps, W0, R0, kpts, True, log, src)
        ST.add(key, meta, mlip_eV=Q.mlip_energy(at, MACE), source=src, ecut=[W0, R0], **info)
ST.finish()
