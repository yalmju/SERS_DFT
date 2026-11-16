# SERS DFT: THI · TBZ · DQ adsorption on Au(111)

## Introduction

This repository contains the DFT workflow used to study competitive adsorption for a CNF@Au SERS ink. Thiram (THI), thiabendazole (TBZ) and diquat (DQ·2Br) are compared on Au(111) at several surface densities, against the cellulose nanofiber (CNF) binder, and in 1:1:1 co-adsorption.

All calculations are Quantum ESPRESSO (pw.x) single-points at the PBE + D3(BJ) level, with D3 computed by QE itself, and SSSP PBE efficiency pseudopotentials. They run on structures relaxed with MACE-MP-0 + D3 (MLIP), and energies are referenced to water using GFN2-xTB/ALPB solvation free energies:

E_ads,aq(X) = E(Au+X) − E(Au) − [E_gas(X) + G_solv(X)]

The workflow covers:
- DFT validation of MLIP adsorption energies at 118 Å² per molecule (4×4 cell)
- a convergence test on cutoff and k-points
- adsorption at other surface densities (3×2, 3×3 and 5×5 cells)
- competition with CNF and displacement from CNF onto Au (6×6 cell)
- re-ranking of 1:1:1 co-adsorption states at 118 Å²

## Dataset

Input structures and results are not included in this repository.

The scripts expect the MLIP and xTB inputs in `data/`:
- `results_all`: gas-phase molecules
- `results_compete`: adsorbed structures per surface density
- `results_fix`: high-density THI structures and solvation energies (`fix_summary.csv`)
- `results_cnf`: CNF and molecules on Au, plus CNF binding energies
- `results_ternary`: 1:1:1 co-adsorption structures in the 4×4 cell

DFT outputs (QE `.in`/`.out` files, CSV tables and logs) are written to `results/`. If `pseudo/` has no SSSP set, `04_dft_singlepoint.py` downloads SSSP 1.3.0 PBE efficiency into it. When several SSSP JSON files are present, the last one in sorted order is used, so keep only the set you intend to use.

# Running the code

Run every script from the repository root on macOS or Linux; the scripts call `mpirun` through a POSIX shell (zsh if available). QE scratch files go to `~/qe_tmp`, so do not run two scripts at the same time.

A finished calculation (`.out` containing `JOB DONE`) is reused instead of rerun. Scripts 07, 12, 15 and 16 reuse it only if the elements, atomic positions and cell match the previous input. They do not compare cutoff, k-points or pseudopotentials, so delete the matching folder in `results/` before rerunning with different settings.

### 1. Create the environment using the file `sers_dft.yml`

```bash
conda env create -f sers_dft.yml
conda activate sers_dft
```

### 2. DFT validation at 118 Å² (also fetches pseudopotentials and gas-phase references)

```bash
python 04_dft_singlepoint.py
```

The other scripts reuse its gas-phase results in `results/dft_check/`, so run it first.

### 3. Convergence test (cutoff 45/360 Ry and k-points 3×3×1)

```bash
python 07_dft_convergence.py
```

### 4. Adsorption at other surface densities

```bash
DFT_COV_CELLS=3x2,3x3 python 12_dft_coverage.py
DFT_COV_CELLS=5x5 DFT_ECUT=45,360 DFT_NK=1 python 12_dft_coverage.py
```

The second line runs the 5×5 cell with lower-memory settings.

### 5. Competition with CNF (6×6)

```bash
DFT_NK=1 python 15_dft_cnf.py
```

### 6. 1:1:1 co-adsorption at 118 Å²

```bash
DFT_NK=1 python 16_dft_ternary118.py
```

Options:

| Variable | Used by | Meaning |
|---|---|---|
| `DFT_NP` | all | number of MPI processes (default 10) |
| `DFT_NK` | 07, 12, 15, 16 | number of k-point pools (04 always uses 2 for slabs) |
| `DFT_ECUT` | 12, 15 | cutoff override, e.g. `45,360` |
| `DFT_LOWMEM` | 07, 12, 15, 16 | minimal Davidson workspace |
| `DFT_COV_CELLS` | 12 | cells to compute, e.g. `3x2,3x3` |
| `DFT_CNF_K` | 15 | k-points instead of Γ-only, e.g. `2,2,1` |
| `CONV_DQ` | 07 | `1` also tests DQ·2Br |
| `DFT_DRY_RUN` | 04 only | `1` writes QE inputs without running pw.x (MACE structure preparation and the pseudopotential download still run) |

`qe_tools.py` holds the shared QE helpers.

# For cite of this work:

The manuscript is in preparation. A citation will be added once it is published.
