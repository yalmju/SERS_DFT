"""
DFT re-ranking of 1:1:1 co-adsorption (118 Å², Au(111) 4×4×3)

MLIP adsorption energies can deviate from DFT (see 04), which may change the ranking of
co-adsorbed states. This recomputes the state energies with DFT (PBE-D3) single-points on the
MLIP co-adsorption structures (all 4×4 states with an MLIP value in ternary.csv).

    G(S) = E(Au + S) − E(Au) − Σ_{X∈S} [E_gas(X) + G_solv(X)]      (water-referenced; computed by aggregate.py)
Gas molecules: 04 structures and box, at the same cutoff.
Option: DFT_ECUT=45,360 (cutoff for slabs and gas references).

Input:  data/results_ternary/, data/results_fix/fix_summary.csv, completed 04 results
Output: results/dft_ternary118/ (QE files, *.meta.json, record.json, ternary118_log.txt); table in results/tables/
"""
import csv, os
import qe_tools as Q
from ase.io import read

KPTS = (2, 2, 1)
ECUT = Q.env_ecut("DFT_ECUT")

ST = Q.Stage("ternary118", "dft_ternary118", "ternary118_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun")
table = Q.load_sssp(log)
MACE = Q.load_mace(log)

T = [r for r in csv.reader(open("data/results_ternary/ternary.csv")) if len(r) >= 7 and r[0] == "4x4" and r[4]]
STATES = {r[2]: float(r[4]) for r in T if os.path.exists(f"data/results_ternary/4x4_{r[2].replace('+', '_')}.traj")}
if not STATES: Q.fail("data/results_ternary/4x4_*.traj not found")
ST.constants.update({"G_solv_eV": Q.read_gsolv(), "mlip_state_eV": STATES,
                     "mlip_state_source": "data/results_ternary/ternary.csv"})

SYS = {k: read(f"data/results_ternary/4x4_{k.replace('+', '_')}.traj") for k in STATES}
SRC = {k: Q.source(f"data/results_ternary/4x4_{k.replace('+', '_')}.traj") for k in STATES}
SYS = {"Au": Q.relax_bare(Q.bare_from(next(iter(SYS.values()))), MACE, log), **SYS}
SRC["Au"] = {"derived_from": next(iter(SRC.values()))["path"], "note": "Au atoms only, MLIP-relaxed"}
Q.align_cells(SYS, vacuum=13.0)

_, W0, R0 = Q.pp_info(table, sorted({e for s in SYS.values() for e in s.get_chemical_symbols()}))
if ECUT: W0, R0 = ECUT
log(f"\n=== DFT single-point, Au(111) 4×4 (118 Å²), k {KPTS}, ecut {W0}/{R0} Ry ===")
for n in ("THI", "TBZ", "DQ"):
    ST.add(f"{n}_gas", Q.gas_reference(n, table, (W0, R0), f"{ST.dir}/gas", log), system=f"{n}_gas", molecule=n, role="gas")
order = ["Au"] + (["THI"] if "THI" in STATES else []) + [k for k in sorted(STATES, key=STATES.get) if k != "THI"]
for key in order:
    at = SYS[key]
    pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
    meta = Q.run_pw(ST.dir, f"S_{key.replace('+', '_')}", at, pps, W0, R0, KPTS, True, log, SRC[key])
    ST.add(key, meta, system=key, role="slab" if key == "Au" else "state",
           molecules=None if key == "Au" else key.split("+"))
ST.finish()
