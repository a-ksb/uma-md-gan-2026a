# Data and code for: Molecular dynamics with a first-principles-validated universal machine-learning potential reveals dynamic elementary processes of growth-related adspecies on GaN(0001)

Y. Takaesu, A. Kusaba, J. Ishii, S. Matsushima, and Y. Kangawa,
arXiv:2607.23461 [cond-mat.mtrl-sci] (2026). https://arxiv.org/abs/2607.23461

This repository contains the simulation input files, analysis scripts, and the
data underlying the figures of the paper.

## Repository structure

```
fpmd_castep/
  settings/
    geomopt_settings_header.txt      # CASTEP output header: structural optimization (LBFGS, PBE, 330 eV)
    md_segment1_settings_header.txt  # CASTEP output header: FPMD segment 1 (NVT, 1273 K, Nose-Hoover, dt = 0.1 fs, 280 eV)
uma_md/
  oc20_relax_then_lammps_nvt_workflow.py  # UMA optimization + LAMMPS (fix external) NVT-MD workflow
  run_oc20_fix_external_b_batch.sh        # batch run script
  01_optimized_t0.vasp                    # UMA-optimized initial structure (t = 0)
  fixed_atom_ids_1based.txt               # IDs of fixed atoms (bottom N layer + terminating H)
  analysis/
    plot_ganh_distance.py                 # Ga-N_admol distance analysis (dissociation/re-association)
    ganh_distance.csv
    compute_ga_topga_dmin.py              # min. Ga_admol-(top-layer Ga) distance extraction
    ga_topga_dmin.csv                     # basis of the bonding-state occupancy analysis (56%/44%)
validation/
  code/                # pipeline for UMA single-point calculations on 751 FPMD snapshots
  outputs/             # DFT/UMA energy tables, parity statistics, structural descriptors (data underlying the validation figures)
  snapshots/           # coordinates of the 751 FPMD snapshots (2 fs interval, 1.5 ps)
```

## Figure provenance

- Fig. 4 (UMA validation, parity by bonding state):
  `validation/code/09_fig_parity_by_state.py` with the data in `validation/outputs/`
- Fig. 5 (dissociation/re-formation time series):
  `uma_md/analysis/plot_ganh_distance.py` with `uma_md/analysis/ganh_distance.csv`
- Figs. 1-3 and 6 (atomic configurations and trajectories): rendered with OVITO
- Bonding-state occupancies in the MLIP-MD run (56%/44%, Ga-Ga threshold 3.3 Å):
  `uma_md/analysis/compute_ga_topga_dmin.py` with `uma_md/analysis/ga_topga_dmin.csv`

## Computational conditions

- **FPMD (CASTEP)**: PBE, ultrasoft pseudopotentials, plane-wave cutoff 280 eV,
  Γ-point sampling, NVT at 1273 K (Nose-Hoover chain), time step 0.1 fs,
  15 segments × 1000 steps = 1.5 ps. Full parameter listings are in
  `fpmd_castep/settings/`.
- **MLIP-MD**: UMA (uma-m-1p1) via fairchem, driven by LAMMPS
  (`fix external`) through ASE; NVT at 1273 K, 150 ps.
  See `uma_md/oc20_relax_then_lammps_nvt_workflow.py` for all settings.

## Requirements

- Python 3.10+, [fairchem](https://github.com/facebookresearch/fairchem) (UMA
  model `uma-m-1p1`), ASE, LAMMPS, NumPy, pandas, Matplotlib

The UMA universal machine-learning interatomic potential is publicly released
by Meta and available through the fairchem repository.

Raw MD trajectory files are not included due to their size (tens of GB); they
are available from the corresponding author on reasonable request.
