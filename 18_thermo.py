"""
Thermal free-energy corrections (zero-point energy, vibrational enthalpy, entropy) from MLIP frequencies

DFT energies are electronic energies. This stage adds, at T = 298.15 K,

    G_ads,aq(X) = E_ads,aq(X) + ΔG_corr(X),   ΔG_corr = G_corr(adsorbed X) − G_corr(X, gas, 1 M)

  adsorbed  ZPE + U_vib − T·S_vib from a partial Hessian of the adsorbate atoms (Au fixed; MACE-MP-0 + D3,
            finite differences ±0.01 Å) after tight MLIP relaxation of the adsorbate (fmax 0.01 eV/Å).
            The Au slab is held fixed, so the bare-slab term cancels.
  gas       ideal gas at 1 atm (translation, rigid rotor, vibrations; σ = 1, singlet) plus RT·ln(24.46)
            for the 1 atm → 1 M standard state. G_solv from xTB/ALPB is a 1 M gas → 1 M solution value.
If the tight relaxation changes the pose (heavy-atom RMSD after superposition > 0.5 Å, height span change
> 1 Å or lowest-atom height change > 0.5 Å), frequencies are taken at the input geometry instead
(imaginary modes then appear; truhlar100 counts them as 100 cm⁻¹, the other schemes drop them).
Low-frequency modes are handled three ways (all stored; the table reports each):
  harmonic     all real modes as harmonic oscillators, imaginary modes dropped (count recorded)
  truhlar100   modes below 100 cm⁻¹ (and imaginary modes) raised to 100 cm⁻¹ for U_vib and S_vib
  grimme       quasi-RRHO entropy (Grimme 2012, ν0 = 100 cm⁻¹, B_av = 1e-44 kg m²), harmonic U_vib
DQ·2Br: the reference is the gas-phase ion pair as one particle; "DQ_dissociated" is also stored
(DQ²⁺ + 2 Br⁻ as three particles at 1 M) to show how sensitive DQ values are to that choice.

Structures: every pose of 17, every 4×4 state of 16, and the CNF-stage structures of 15.
No pw.x is run. Results are cached per input file (SHA-256), so an interrupted run resumes.
Option: DFT_THERMO_THREADS (torch threads, default 2; run with `nice` next to pw.x).

Input:  data/results_compete/, data/results_fix/, data/results_ternary/, data/results_cnf/, completed 04 results
Output: results/thermo/ (thermo.json, vib/ finite-difference cache, thermo_log.txt); tables in results/tables/
"""
import glob, json, os, time
import numpy as np
import qe_tools as Q
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.units import kB, invcm, _hplanck, _k, _c, _amu

T = 298.15
NU0 = 100.0           # cm-1, quasi-RRHO threshold
B_AV = 1e-44          # kg m^2, Grimme average moment of inertia
STD_1M = kB * T * np.log(24.465)   # 1 atm ideal gas -> 1 M (eV)
SETTINGS = {"T_K": T, "nu0_cm1": NU0, "fmax_ads": 0.01, "fmax_gas": 0.005, "delta_A": 0.01, "nfree": 2,
            "mlip": Q.MACE_MODEL["name"] + " + D3(BJ)"}

POSE = {"internal_rmsd_A": 0.5, "dz_span_A": 1.0, "dz_lowest_A": 0.5}   # pose-change thresholds

import torch
torch.set_num_threads(Q._env_int("DFT_THERMO_THREADS", 2))
OUT = os.path.join(Q.RESULTS, "thermo")
os.makedirs(os.path.join(OUT, "vib"), exist_ok=True)
LOG = open(os.path.join(OUT, "thermo_log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ---------------------------------------------------------------- thermochemistry (eV, K)
def _vib(nu):
    """ZPE, U_vib (thermal part), S_vib for real frequencies nu (cm-1)."""
    nu = np.asarray([x for x in nu if x > 0], float)
    e = nu * invcm; x = e / (kB * T)
    zpe = 0.5 * e.sum()
    u = (e / np.expm1(x)).sum()
    s = (kB * (x / np.expm1(x) - np.log(-np.expm1(-x)))).sum()
    return zpe, u, s


def _s_rotor(nu):
    """Free-rotor entropy of Grimme's quasi-RRHO for frequencies nu (cm-1)."""
    nu = np.asarray(nu, float)
    mu = _hplanck / (8 * np.pi ** 2 * nu * 100 * _c)
    mu_eff = mu * B_AV / (mu + B_AV)
    return kB * (0.5 + np.log(np.sqrt(8 * np.pi ** 3 * mu_eff * _k * T / _hplanck ** 2)))


def corrections(freqs):
    """{scheme: {ZPE, U, S, G}} for vibrational modes only (eV, eV/K). freqs: cm-1, imaginary as negative."""
    f = np.asarray(freqs, float); real = f[f > 0]
    out = {}
    zpe, u, s = _vib(real)
    out["harmonic"] = dict(ZPE=zpe, U=u, S=s)
    raised = np.maximum(np.abs(f), NU0)
    _, u2, s2 = _vib(raised)
    out["truhlar100"] = dict(ZPE=zpe, U=u2, S=s2)
    if len(real):
        w = 1.0 / (1.0 + (NU0 / real) ** 4)
        x = real * invcm / (kB * T)
        sv = kB * (x / np.expm1(x) - np.log(-np.expm1(-x)))
        s3 = float((w * sv + (1 - w) * _s_rotor(real)).sum())
    else:
        s3 = 0.0
    out["grimme"] = dict(ZPE=zpe, U=u, S=s3)
    for v in out.values(): v["G"] = v["ZPE"] + v["U"] - T * v["S"]
    return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in out.items()}


def gas_terms(atoms):
    """Translational + rotational U and S of an ideal gas at 1 atm (nonlinear rigid rotor, σ = 1)."""
    m = atoms.get_masses().sum() * _amu
    p = 101325.0
    s_tr = kB * (np.log((2 * np.pi * m * _k * T / _hplanck ** 2) ** 1.5 * _k * T / p) + 2.5)
    out = dict(U_trans=1.5 * kB * T, S_trans=float(s_tr), U_rot=0.0, S_rot=0.0, pV=kB * T)
    if len(atoms) > 1:
        I = atoms.get_moments_of_inertia() * _amu * 1e-20          # kg m^2
        s_rot = kB * (np.log(np.sqrt(np.pi * np.prod(I)) * (8 * np.pi ** 2 * _k * T / _hplanck ** 2) ** 1.5) + 1.5)
        out.update(U_rot=1.5 * kB * T, S_rot=float(s_rot))
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------- MLIP relaxation and frequencies
MACE = Q.load_mace(log)


def pose_change(a, p0, idx, gas):
    """Rigid-body-free change of the molecule: heavy-atom RMSD after optimal superposition, and for
    adsorbates the change in heavy-atom height span and lowest heavy-atom height."""
    hv = [i for i in idx if a[i].symbol != "H"] or idx
    P, R = a.positions[hv] - a.positions[hv].mean(0), p0[hv] - p0[hv].mean(0)
    U, _, Vt = np.linalg.svd(P.T @ R); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    out = {"internal_rmsd_A": float(np.sqrt((((P @ (U @ D @ Vt)) - R) ** 2).sum(1).mean()))}
    if not gas:
        out["dz_span_A"] = float(np.ptp(a.positions[hv, 2]) - np.ptp(p0[hv, 2]))
        out["dz_lowest_A"] = float(a.positions[hv, 2].min() - p0[hv, 2].min())
    return out


def frequencies(atoms, label, gas):
    """Tight MLIP relaxation, then finite-difference frequencies (cm-1, imaginary as negative).
    Returns (atoms used, frequencies, relaxation summary)."""
    a = atoms.copy(); a.set_constraint(); a.calc = MACE
    if gas:
        a.pbc = False; idx = list(range(len(a)))
    else:
        au = [i for i, x in enumerate(a) if x.symbol == "Au"]
        idx = [i for i in range(len(a)) if i not in au]
        a.set_constraint(FixAtoms(indices=au))
    p0 = a.positions.copy()
    fmax = SETTINGS["fmax_gas"] if gas else SETTINGS["fmax_ads"]
    opt = BFGS(a, logfile=None, maxstep=0.1); ok = bool(opt.run(fmax=fmax, steps=3000))
    rel = dict(converged=ok, steps=opt.nsteps, max_shift_A=float(np.abs(a.positions - p0).max()),
               fmax_final=float(np.linalg.norm(a.get_forces()[idx], axis=1).max()), **pose_change(a, p0, idx, gas))
    rel["pose_kept"] = rel["internal_rmsd_A"] <= POSE["internal_rmsd_A"] and (gas or (
        abs(rel["dz_span_A"]) <= POSE["dz_span_A"] and abs(rel["dz_lowest_A"]) <= POSE["dz_lowest_A"]))
    tag = ""
    if not rel["pose_kept"]:   # the relaxation left the pose of the DFT structure: use the input geometry
        a.positions = p0; tag = "__input"
    rel["frequencies_at"] = "relaxed" if rel["pose_kept"] else "input geometry"
    vd = os.path.join(OUT, "vib", label + tag)
    vib = Vibrations(a, indices=idx, delta=SETTINGS["delta_A"], nfree=SETTINGS["nfree"], name=vd)
    vib.run()
    f = vib.get_frequencies()
    freqs = [float(x.real) if abs(x.imag) < 1e-8 else -float(abs(x.imag)) for x in f]
    return a, freqs, rel


def structures():
    """label -> (path, role, molecules) for every structure DFT stages use."""
    S = {}
    for f in sorted(glob.glob("data/results_compete/*_*_*.traj")):
        n = os.path.basename(f).split("_")[0]; S[f] = ("adsorbed", [n])
    for f in sorted(glob.glob("data/results_fix/THI_DMDTCx2_*.traj")): S[f] = ("adsorbed", ["THI"])
    for f in sorted(glob.glob("data/results_ternary/4x4_*.traj")):
        S[f] = ("adsorbed", os.path.basename(f)[4:-5].split("_"))
    for m in ("CNF", "THI", "TBZ", "DQ"):
        f = f"data/results_cnf/best_{m}.traj"
        if os.path.exists(f): S[f] = ("adsorbed", [m])
    return S


path = os.path.join(OUT, "thermo.json")
DB = json.load(open(path)) if os.path.exists(path) else {}
if DB.get("settings") != SETTINGS: DB = {"settings": SETTINGS, "structures": {}, "gas": {}}
DB.update(schema_version=Q.SCHEMA_VERSION, std_state_1M_eV=float(STD_1M))


def save():
    if not Q.DRY: json.dump(DB, open(path, "w"), indent=1)


log(f"\n[thermo] {time.strftime('%Y-%m-%d %H:%M:%S')}  mode {Q.MODE}  T {T} K  threads {torch.get_num_threads()}")

# gas molecules: 04 structures (the DFT gas references), CNF from data/results_cnf
S04 = Q.structures_from_04()
GAS = {n: S04[f"{n}_gas"] for n in ("THI", "TBZ", "DQ") if f"{n}_gas" in S04}
if os.path.exists("data/results_cnf/CNF_gas.xyz"): GAS["CNF"] = read("data/results_cnf/CNF_gas.xyz")
if "DQ" in GAS:
    GAS["DQ2+"] = GAS["DQ"][[a.index for a in GAS["DQ"] if a.symbol != "Br"]]
for n, at in GAS.items():
    h = Q.fp_hash({"symbols": at.get_chemical_symbols(), "positions": np.round(at.positions, 5).tolist()})
    if DB["gas"].get(n, {}).get("input_hash") == h and "pose_kept" in DB["gas"][n].get("relax", {}):
        log(f"  ✓ gas {n:<6s} cached"); continue
    if Q.DRY:
        log(f"  [dry-run] gas {n}"); continue
    t = time.time()
    a, fr, rel = frequencies(at, f"gas_{n.replace('+', 'p')}", gas=True)
    vib = sorted(fr, key=abs)[6:]                     # drop 3 translations + 3 rotations
    c = corrections(vib); g = gas_terms(a)
    for v in c.values():
        v["G_1atm"] = v["G"] + g["U_trans"] + g["U_rot"] + g["pV"] - T * (g["S_trans"] + g["S_rot"])
        v["G_1M"] = v["G_1atm"] + STD_1M
    DB["gas"][n] = dict(input_hash=h, freqs_cm1=fr, n_imag=int(sum(x < 0 for x in vib)), relax=rel, ideal_gas=g, corr=c)
    save()
    log(f"  ▶ gas {n:<6s} G_corr(1 M, truhlar100) {c['truhlar100']['G_1M']:+.3f} eV  "
        f"({len(vib)} modes, {DB['gas'][n]['n_imag']} imaginary, {time.time() - t:.0f}s)")
if "DQ" in GAS:   # Br- for the dissociated DQ reference: translation only
    g = gas_terms(GAS["DQ"][[a.index for a in GAS["DQ"] if a.symbol == "Br"]][:1])
    DB["gas"]["Br-"] = dict(ideal_gas=g, corr={k: {"G_1M": g["U_trans"] + g["pV"] - T * g["S_trans"] + STD_1M}
                                             for k in ("harmonic", "truhlar100", "grimme")})

for f, (role, mols) in structures().items():
    h = Q.sha256(f)
    if DB["structures"].get(f, {}).get("sha256") == h and "pose_kept" in DB["structures"][f].get("relax", {}):
        log(f"  ✓ {f:<46s} cached"); continue
    if Q.DRY:
        log(f"  [dry-run] {f}"); continue
    t = time.time()
    label = f.replace("data/", "").replace("/", "__")[:-5]
    a, fr, rel = frequencies(read(f), label, gas=False)
    c = corrections(fr)
    DB["structures"][f] = dict(sha256=h, molecules=mols, n_ads_atoms=sum(x.symbol != "Au" for x in a),
                               freqs_cm1=fr, n_imag=int(sum(x < 0 for x in fr)), relax=rel, corr=c)
    save()
    log(f"  ▶ {f:<46s} G_corr(truhlar100) {c['truhlar100']['G']:+.3f} eV  ({len(fr)} modes, "
        f"{DB['structures'][f]['n_imag']} imaginary, relax {'ok' if rel['converged'] else 'NOT converged'}, "
        f"internal RMSD {rel['internal_rmsd_A']:.2f} Å, {'pose kept' if rel['pose_kept'] else 'POSE CHANGED → input geometry'}, "
        f"{time.time() - t:.0f}s)")

save()
if not Q.DRY:
    import aggregate
    aggregate.build(Q.RESULTS, log=log)
log(f"[thermo] done {time.strftime('%H:%M:%S')}")
