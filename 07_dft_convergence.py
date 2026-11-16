"""
DFT convergence test: sensitivity of ΔE(THI←TBZ) to cutoff and k-points (118 Å², 4×4, 04 structures)

  ref   : 04 cutoff (SSSP recommendation), k 2×2×1   (identical to 04, reused)
  ecut45: ecutwfc 45 / ecutrho 360 Ry,     k 2×2×1   (Au, THI@Au, TBZ@Au, THI(g), TBZ(g))
  k3    : 04 cutoff,                       k 3×3×1   (Au, THI@Au, TBZ@Au; gas molecules identical to ref)
  CONV_DQ=1 also includes DQ·2Br

Input:  completed 04 results (results/dft_check/), data/results_fix/fix_summary.csv
Output: results/dft_convergence/ (QE files, *.meta.json, record.json, convergence_log.txt); table in results/tables/
"""
import qe_tools as Q

ST = Q.Stage("convergence", "dft_convergence", "convergence_log.txt")
log = ST.log
Q.check_programs("pw.x", "mpirun")
table = Q.load_sssp(log)
S = Q.structures_from_04()
MOLS = ["THI", "TBZ"] + (["DQ"] if Q.env_flag("CONV_DQ") else [])
ref = Q.read_meta("results/dft_check", "Au")
W0, R0 = (ref["fingerprint"]["ecutwfc"], ref["fingerprint"]["ecutrho"]) if ref else (60.0, 480.0)
ST.constants["G_solv_eV"] = Q.read_gsolv()
ST.constants["molecules"] = MOLS
SETTINGS = {"ref": ((W0, R0), (2, 2, 1)), "ecut45": ((45.0, 360.0), (2, 2, 1)), "k3": ((W0, R0), (3, 3, 1))}
ST.constants["settings"] = {k: {"ecut": list(e), "kpts": list(kp)} for k, (e, kp) in SETTINGS.items()}

for sname, ((w, r), kp) in SETTINGS.items():
    log(f"\n=== {sname}: ecut {w}/{r} Ry, k {kp} ===")
    for key in ["Au"] + [f"{m}@Au" for m in MOLS] + [f"{m}_gas" for m in MOLS]:
        mol = None if key == "Au" else key.split("@")[0].split("_")[0]
        role = "slab" if key == "Au" else "adsorbed" if "@" in key else "gas"
        if key not in S:
            ST.add(f"{sname}/{key}", None, "not_available", "04 structure not completed",
                   setting=sname, system=key, molecule=mol, role=role); continue
        at = S[key]; slab = role != "gas"
        pps, _, _ = Q.pp_info(table, sorted(set(at.get_chemical_symbols())))
        meta = Q.run_or_reuse(f"{ST.dir}/{sname}", key.replace("@", "_"), at, pps, w, r, kp if slab else None,
                              slab, log, {"path": f"results/dft_check/{key.replace('@', '_')}.in"})
        ST.add(f"{sname}/{key}", meta, setting=sname, system=key, molecule=mol, role=role)
ST.finish()
