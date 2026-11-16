"""
Shared Quantum ESPRESSO helpers (used by 07, 12, 15, 16)

Same settings as 04_dft_singlepoint.py:
  PBE + D3(BJ) (dftd3_version=4), SSSP PBE efficiency, Marzari–Vanderbilt 0.01 Ry, conv_thr 1e-6 Ry
  Slabs: Davidson + local-TF mixing; gas molecules: CG (avoids Γ-point Davidson numerical errors)
  mpirun -np $DFT_NP pw.x -nd 1  (parallel diagonalization off: avoids segfaults in the macOS conda build)
"""
import os
import numpy as np, sys, glob, json, shutil, subprocess, time
from ase.io import read, write

NP = int(os.environ.get("DFT_NP", 10))
PSEUDO = os.path.abspath("pseudo")
TMP = os.path.expanduser("~/qe_tmp")          # QE scratch directory
os.makedirs(TMP, exist_ok=True)
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           OMPI_MCA_rmaps_base_oversubscribe="1")
SHELL = "/bin/zsh" if os.path.exists("/bin/zsh") else None


def check_programs(*exes):
    for exe in exes:
        if shutil.which(exe) is None:
            sys.exit(f"[abort] {exe} not found; conda install -c conda-forge -y qe openmpi")


def load_sssp():
    jf = sorted(glob.glob(f"{PSEUDO}/**/SSSP_*PBE*efficiency*.json", recursive=True))
    if not jf:
        sys.exit("[abort] no SSSP json in pseudo/ (run 04_dft_singlepoint.py first).")
    return json.load(open(jf[-1])), os.path.basename(jf[-1])


def pp_info(sssp, elements):
    pps, wfc, rho = {}, 0, 0
    for el in elements:
        d = sssp[el]; fn = d["filename"]
        hits = glob.glob(f"{PSEUDO}/**/{fn}", recursive=True)
        if not hits: sys.exit(f"[abort] {fn} not found in pseudo/.")
        pps[el] = os.path.relpath(hits[0], PSEUDO)
        cw = d.get("cutoff_wfc", d.get("cutoff")); cr = d.get("cutoff_rho", cw * d.get("dual", 8))
        wfc, rho = max(wfc, cw), max(rho, cr)
    return pps, wfc, rho


def mode_d3():
    """Run mode fixed by 04: (diagonalization, QE built-in D3). Only QE built-in D3 is supported."""
    f = "results/dft_check/mode.json"
    mode = tuple(json.load(open(f))) if os.path.exists(f) else ("david", True)
    if not mode[1]:
        sys.exit(f"[abort] {f} requests D3 without QE built-in D3, which is not supported.")
    return mode


def write_input(path, atoms, prefix, pps, ecutwfc, ecutrho, kpts, slab, diag, d3=True):
    data = {
        "control": {"calculation": "scf", "prefix": prefix, "pseudo_dir": PSEUDO,
                    "outdir": os.path.join(TMP, prefix), "disk_io": "low", "tprnfor": False},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "mv", "degauss": 0.01},
        "electrons": {"conv_thr": 1e-6, "diagonalization": diag,
                      "mixing_beta": 0.2 if slab else 0.4,
                      "mixing_mode": "local-TF" if slab else "plain", "electron_maxstep": 300},
    }
    if d3: data["system"].update({"vdw_corr": "dft-d3", "dftd3_version": 4})
    if os.environ.get("DFT_LOWMEM"):            # save memory: minimal Davidson workspace
        data["electrons"]["diago_david_ndim"] = 2
    write(path, atoms, format="espresso-in", input_data=data, pseudopotentials=pps, kpts=kpts)
    txt = open(path).read()
    for nl in ("&FCP\n/\n", "&RISM\n/\n", "&CELL\n/\n", "&IONS\n/\n"):
        txt = txt.replace(nl, "")
    open(path, "w").write(txt)


def done(out):
    return os.path.exists(out) and "JOB DONE" in open(out, errors="ignore").read()


def energy(out):
    return read(out, format="espresso-out").get_potential_energy()


def run_pw(workdir, name, atoms, pps, ecutwfc, ecutrho, kpts, slab, log):
    """Skip if already done; on failure, retry with other diagonalization/k-point setups. Returns the output path or None."""
    os.makedirs(workdir, exist_ok=True)
    fout = os.path.join(workdir, f"{name}.out")
    if done(fout):
        same = False
        try:                                            # reuse only if elements, positions and cell match
            prev = read(os.path.join(workdir, f"{name}.in"), format="espresso-in")
            same = (prev.get_chemical_symbols() == atoms.get_chemical_symbols()
                    and np.abs(prev.positions - atoms.positions).max() < 1e-3
                    and np.abs(prev.cell.array - atoms.cell.array).max() < 1e-3)
        except Exception:                               # unreadable input: do not reuse
            pass
        if same:
            log(f"  ✓ {name:<22s} (reusing existing result)"); return fout
        log(f"  ↻ {name:<22s} structure changed or previous input unreadable, recomputing")
    diag0, d3 = mode_d3()
    if slab:
        # Γ-only (gamma trick) + Davidson can fail on large slabs (6×6),
        # so try regular k=1×1×1 Davidson before CG (very slow)
        tries = [(diag0, kpts)] + ([(diag0, (1, 1, 1))] if kpts is None else []) + [("cg", kpts)]
    else:
        tries = [("cg", kpts), (diag0, kpts), ("cg", (1, 1, 1))]
    for k, (diag, kp) in enumerate(tries, 1):
        fin = os.path.join(workdir, f"{name}.in")
        write_input(fin, atoms, name, pps, ecutwfc, ecutrho, kp, slab, diag, d3)
        shutil.rmtree(os.path.join(TMP, name), ignore_errors=True)
        os.makedirs(os.path.join(TMP, name), exist_ok=True)
        nk = 2 if (slab and kp and kp[0] * kp[1] > 1) else 1
        nk = int(os.environ.get("DFT_NK", nk))          # DFT_NK=1: no k-point pools, about half the memory
        cmd = (f"ulimit -s hard 2>/dev/null; mpirun -np {NP} pw.x -nk {nk} -nd 1 "
               f"-in {name}.in > {name}.out 2> {name}.err")
        t = time.time()
        log(f"  ▶ {name:<22s} start {time.strftime('%H:%M')} (ecut {ecutwfc}/{ecutrho} Ry, k {kp or 'Γ'}, {diag})")
        subprocess.run(cmd, shell=True, cwd=workdir, env=ENV, executable=SHELL)
        if done(fout):
            log(f"    done ({(time.time()-t)/60:.1f} min)")
            shutil.rmtree(os.path.join(TMP, name), ignore_errors=True)
            return fout
        log(f"    failed (attempt {k}){' → retrying' if k < len(tries) else ''}")
        for ext in ("out", "err"):                       # keep for diagnosing the failure
            src = os.path.join(workdir, f"{name}.{ext}")
            if os.path.exists(src): shutil.copy(src, os.path.join(workdir, f"{name}.fail{k}.{ext}"))
        if os.path.exists(fout):
            tail = [l.strip() for l in open(fout, errors="ignore").read().splitlines() if "Error" in l or "%%%%" in l]
            errt = open(os.path.join(workdir, f"{name}.err"), errors="ignore").read().strip().splitlines()[-3:]
            log("      cause: " + (" | ".join(tail[:3] + errt) or "no output (possibly out of memory or killed)"))
    shutil.rmtree(os.path.join(TMP, name), ignore_errors=True)
    return None



def read_structures_from_singlepoint():
    """Read the structures (input files) actually computed by 04_dft_singlepoint.py, for further runs on the same geometry"""
    S = {}
    for key, f in [("Au", "Au.in"), ("THI@Au", "THI_Au.in"), ("TBZ@Au", "TBZ_Au.in"), ("DQ@Au", "DQ_Au.in"),
                   ("THI_gas", "THI_gas.in"), ("TBZ_gas", "TBZ_gas.in"), ("DQ_gas", "DQ_gas.in")]:
        p = os.path.join("results/dft_check", f)
        if os.path.exists(p):
            S[key] = read(p, format="espresso-in")
    if "Au" not in S:
        sys.exit("[abort] results/dft_check/*.in not found. Run 04_dft_singlepoint.py first.")
    return S


def ecut_override():
    """DFT_ECUT=45,360 → (45, 360). Cutoff checked against 60/480 in the 07 convergence test"""
    v = os.environ.get("DFT_ECUT")
    return tuple(float(x) for x in v.split(",")) if v else None


def gas_energy(name, ecut, workdir, pps, log):
    """Gas-phase DFT energy at the requested cutoff. ecut=None: 04 result (dft_check) as is.
    Other cutoffs: reuse a 07 result at that cutoff if present, else recompute with the 04 structure and box."""
    if ecut is None:
        return energy(f"results/dft_check/{name}_gas.out")
    old = f"results/dft_convergence/ecut{int(ecut[0])}/{name}_gas.out"
    if done(old):
        log(f"  ✓ {name}_gas ({int(ecut[0])}/{int(ecut[1])} Ry, reused from 07)"); return energy(old)
    at = read(f"results/dft_check/{name}_gas.in", format="espresso-in")
    out = run_pw(workdir, f"{name}_gas", at, pps, ecut[0], ecut[1], None, False, log)
    return energy(out) if out else None
