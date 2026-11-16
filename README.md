# SERS DFT: THI · TBZ · DQ adsorption on Au(111)

## Introduction

This repository contains the DFT workflow used to study competitive adsorption for a CNF@Au SERS ink. Thiram (THI), thiabendazole (TBZ) and diquat (DQ·2Br) are compared on Au(111) at several surface densities, against the cellulose nanofiber (CNF) binder, and in 1:1:1 co-adsorption.

All calculations are Quantum ESPRESSO (pw.x) single-points at the PBE + D3(BJ) level, with D3 computed by QE itself, using the SSSP 1.0 PBE efficiency pseudopotentials. They run on structures relaxed with MACE-MP-0 + D3 (MLIP), and energies are referenced to water using GFN2-xTB/ALPB solvation free energies:

E_ads,aq(X) = E(Au+X) − E(Au) − [E_gas(X) + G_solv(X)]

The workflow covers:
- DFT validation of MLIP adsorption energies at 118 Å² per molecule (4×4 cell), `04_dft_singlepoint.py`
- a convergence test on cutoff and k-points, `07_dft_convergence.py`
- adsorption at other surface densities (3×2, 3×3 and 5×5 cells), `12_dft_coverage.py`
- competition with CNF and displacement from CNF onto Au (6×6 cell), `15_dft_cnf.py`
- re-ranking of 1:1:1 co-adsorption states at 118 Å², `16_dft_ternary118.py`

`qe_tools.py` runs and checks every pw.x calculation; `aggregate.py` builds all result tables from stored outputs.

## Dataset

Input structures and results are not included in this repository.

The scripts expect the MLIP and xTB inputs in `data/`:
- `results_all`: gas-phase molecules
- `results_compete`: adsorbed structures per surface density
- `results_fix`: high-density THI structures and solvation energies (`fix_summary.csv`)
- `results_cnf`: CNF and molecules on Au, plus CNF binding energies and footprints
- `results_ternary`: 1:1:1 co-adsorption structures in the 4×4 cell

Outputs go to `results/`:
- `results/<stage>/`: QE `.in`/`.out`/`.err`, one `<name>.meta.json` per calculation, `record.json` (every planned calculation of the stage) and a log
- `results/<stage>/_archive/`: older runs and failed attempts (nothing is deleted)
- `results/tables/`: `calculations.csv`, `check.csv`, `convergence.csv`, `coverage.csv`, `cnf.csv`, `ternary118.csv` and `schema.json`

### Pinned inputs

| Item | Version | Check |
|---|---|---|
| Pseudopotentials | SSSP 1.0 PBE efficiency ([doi:10.24435/materialscloud:2018.0001/v4](https://doi.org/10.24435/materialscloud:2018.0001/v4)) | MD5 of the archive, the JSON and every UPF file |
| MLIP | MACE-MP-0 medium, `2023-12-03-mace-128-L1_epoch-199.model` | SHA-256 |
| Software | Quantum ESPRESSO 7.6, xTB 6.7.1, ASE 3.29.0, NumPy 2.4.6, PyTorch 2.14.1, mace-torch 0.3.17, torch-dftd 0.5.3 | recorded in every `.meta.json` |

`04_dft_singlepoint.py` downloads the SSSP set into `pseudo/` and the MACE model into `~/.cache/mace/` if they are missing. A checksum mismatch stops the run.

# Running the code

Validated on macOS (arm64). Linux should work but has not been tested. The scripts call `mpirun` through a POSIX shell (`zsh` if available, otherwise `sh`) and use `ulimit`. Run every script from the repository root, one at a time.

### 1. Create the environment using the file `sers_dft.yml`

```bash
conda env create -f sers_dft.yml
conda activate sers_dft
```

To reproduce the exact validated macOS arm64 environment instead:

```bash
conda create -n sers_dft --file env/conda-osx-arm64.lock
conda activate sers_dft
pip install -r env/pip-osx-arm64.txt
```

### 2. Check the installation

```bash
python -m unittest discover -s tests -v
```

`test_workflow.py` checks reuse, dry-run, D3 handling, failure handling, aggregation and option checks with a stand-in `pw.x` and runs in seconds. `test_env_smoke.py` runs one tiny real `pw.x` calculation and is skipped until `pseudo/` holds the SSSP set (after step 4 or a manual download).

### 3. Prepare inputs without running DFT (optional)

```bash
DFT_DRY_RUN=1 python 04_dft_singlepoint.py
```

A dry run writes QE inputs and `record.json` to `results_dryrun/` and never calls `pw.x` or `xtb`. It still prepares structures with MACE and may download the pseudopotentials and the MACE model. `results/` is not touched. Stages 07–16 read the 04 structures from `results/dft_check/`, so dry-run them after 04 has completed.

### 4. Run the calculations

```bash
python 04_dft_singlepoint.py
python 07_dft_convergence.py
DFT_COV_CELLS=3x2,3x3 python 12_dft_coverage.py
DFT_COV_CELLS=5x5 DFT_ECUT=45,360 DFT_NK=1 python 12_dft_coverage.py
DFT_NK=1 python 15_dft_cnf.py
DFT_NK=1 python 16_dft_ternary118.py
```

Run 04 first; the other stages reuse its structures and gas-phase references. The second 12 line runs the 5×5 cell with lower-memory settings; partial runs add to the stage record instead of replacing it.

Each stage ends by rebuilding `results/tables/` and exits with status 1 if any calculation failed or is incomplete. Completion requires pw.x exit code 0, `JOB DONE`, a converged SCF, the QE D3 term and a parsed energy. A calculation is reused only if its metadata says completed and its fingerprint matches: elements, positions, cell, periodicity, cutoff, k-points, physical QE settings and UPF checksums. Otherwise the old files are moved to `_archive/` and pw.x runs again.

### 5. Rebuild the tables from stored outputs

```bash
python aggregate.py
```

This runs no calculation. Each table row carries a `status` (`completed`, `failed`, `incomplete`, `not_available`; older runs are `superseded`) and a `reason`. A value is filled only when every input is completed. Missing values are left empty and never treated as zero.

### 6. Recompute or adopt existing outputs

```bash
DFT_RECOMPUTE=1 python 15_dft_cnf.py
DFT_ADOPT=1 python 04_dft_singlepoint.py
```

`DFT_RECOMPUTE=1` archives matching results and runs them again. `DFT_ADOPT=1` attaches metadata to outputs produced before this metadata existed. It compares each stored input with the planned calculation (elements, positions, cell, cutoff, k-points, pseudopotential file names, QE D3) and the output with the completion rules. pw.x is not run. Adopted calculations are marked in `.meta.json`, with run-time checksums and versions listed as unverified.

### Options

| Variable | Used by | Meaning |
|---|---|---|
| `DFT_NP` | all | number of MPI processes (default 10) |
| `DFT_NK` | all | k-point pools (default 2 for slabs with more than one k-point, else 1) |
| `DFT_LOWMEM` | all | `1`: minimal Davidson workspace |
| `DFT_DRY_RUN` | all | `1`: write inputs to `results_dryrun/` only |
| `DFT_RECOMPUTE` | all | `1`: archive and recompute |
| `DFT_ADOPT` | all | `1`: adopt verified pre-existing outputs |
| `DFT_ECUT` | 12, 16 | cutoff `ecutwfc,ecutrho` in Ry for slabs and gas references |
| `DFT_COV_CELLS` | 12 | cells to compute: `3x2`, `3x3`, `5x5` |
| `DFT_CNF_ECUT` | 15 | cutoff (default `45,360`) |
| `DFT_CNF_K` | 15 | k-points `n1,n2,n3` instead of Γ-only |
| `CONV_DQ` | 07 | `1`: also test DQ·2Br |

Each option is checked for format and range, an unknown `DFT_*` variable stops the run, and the options in effect are written to the log. `DFT_DRY_RUN`, `DFT_ADOPT` and `DFT_RECOMPUTE` cannot be combined.

### Resources

Every `.meta.json` records wall time per attempt, MPI and pool settings, the host (CPU, memory) and QE's "Estimated max dynamical RAM". The RAM figure is QE's estimate, not a measured peak. `results/tables/calculations.csv` lists these per calculation. QE scratch files go to a separate folder per run under `~/qe_tmp`.

# For cite of this work:

The manuscript is in preparation. A citation will be added once it is published.

# License

MIT, see `LICENSE`.
