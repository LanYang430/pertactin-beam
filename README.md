# pertactin-beam

Code accompanying the manuscript on pertactin passenger domain folding,
implementing the BEAM (Biophysical Enhanced Adaptive Methods) workflow:
coarse-grained (CG) tICA-guided adaptive sampling (REAP), all-atom (AA)
replica-exchange umbrella sampling (REUS) with WHAM free energy analysis,
and Gillespie kinetic Monte Carlo simulation on the resulting free energy
surface.

This repository contains the analysis and pipeline scripts used to
produce the results in the paper. Simulation input files, trained models,
reduced trajectories, and free energy surfaces are archived separately on
Zenodo: **[DOI/link to be added]**.

## Repository structure

```
pertactin-beam/
├── configs/
│   └── reap_parameters.yaml       REAP adaptive sampling parameters
└── scripts/
    ├── cg_tica/                   Stage 1: CG trajectory -> tICA model
    │   ├── 01_align_cg_trajectories.py
    │   ├── 02_lagtime_scan.py
    │   └── 03_train_cg_tica.py
    ├── reap/                      Stage 2: AA -> CG-tICA projection for REAP
    │   └── project_aa_to_cg_tica.py
    └── analysis/                  Stage 3: PMF, kinetics, and figure analysis
        ├── anisotropy.py
        ├── nonnative_contacts.py
        ├── pmf_convergence.py
        └── gillespie_kinetics.py
        └── pmf_masking.py
```

## Workflow overview

1. **Coarse-grained simulation** was run with
   [Upside](https://github.com/sosnicklab/upside2-md) (external software,
   not included here). See the manuscript Methods for topology, force
   field, and sampling parameters.

2. **CG trajectory alignment and tICA training**
   (`scripts/cg_tica/`): align CG trajectories to a reference structure,
   scan TICA lag times to justify the final lag time, and train the final
   CG tICA model used to define collective variables for REAP.

3. **REAP adaptive sampling**: after each round of all-atom simulations,
   trajectories were projected into the CG-trained tICA CV space using
   `scripts/reap/project_aa_to_cg_tica.py`. The resulting `input.npy`
   files were used by the REAP workflow to select new starting
   structures. REAP used k-means clustering with `k = 2000`, retained the
   top 200 candidate structures, and selected 10 starting structures for
   each new sampling round (`configs/reap_parameters.yaml`).

   Cluster-specific SLURM submission scripts are not included, since they
   depend on the local HPC environment and directory layout.

4. **All-atom REUS and WHAM**: replica-exchange umbrella sampling was run
   in NAMD (external software), and the resulting 2D free energy surface
   (PMF) was computed with the WHAM program
   ([Grossfield lab](http://membrane.urmc.rochester.edu/?page_id=126)).
   `scripts/analysis/pmf_convergence.py` checks PMF convergence as a
   function of per-window simulation time by re-running WHAM on
   truncated trajectories and comparing to the full-trajectory PMF.

5. **Structural analysis on the AA-trained tICA CV space**
   (`scripts/analysis/`): `anisotropy.py` computes the shape anisotropy
   of the unfolded region as a function of tIC1/tIC2; `nonnative_contacts.py`
   computes non-native contact counts within the simulated segment by
   state (A/B/C).

6. **Kinetic Monte Carlo (Gillespie) simulation**
   (`scripts/analysis/gillespie_kinetics.py`): builds a discretized
   kinetic network on the 2D PMF grid using position-dependent diffusion
   coefficients, and runs Gillespie trajectories from an extended/unfolded
   region to the folded basin, following a discretized Smoluchowski rate
   formula (see manuscript Methods, Eq. 2). Supports comparing a
   trap-involved condition against a trap-blocked condition (a region of
   the PMF masked to simulate geometric exclusion).

## External dependencies (not pip-installable)

- [Upside](https://github.com/sosnicklab/upside2-md) - CG molecular dynamics
- NAMD - AA molecular dynamics and REUS
- [WHAM](http://membrane.urmc.rochester.edu/?page_id=126) (`wham-2d`) -
  free energy analysis; must be available on `PATH` for
  `pmf_convergence.py`

## Python dependencies

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Usage

Each script accepts `--help` for full argument documentation. Example
invocations:

```bash
# Stage 1: CG trajectory alignment and tICA training
python scripts/cg_tica/01_align_cg_trajectories.py input.dcd aligned.dcd \
    --topology protein_upside.pdb

python scripts/cg_tica/02_lagtime_scan.py --pdb protein_upside.pdb \
    --traj_glob "cg_aligned/*.dcd"

python scripts/cg_tica/03_train_cg_tica.py --pdb protein_upside.pdb \
    --traj_glob "cg_aligned/*.dcd" --out cg_tica_model.pkl

# Stage 2: project an AA trajectory for REAP
python scripts/reap/project_aa_to_cg_tica.py aa_traj.dcd aa_topology.pdb \
    --reference reference_all_atom.pdb --model cg_tica_model.pkl \
    --out input.npy

# Stage 3: PMF convergence
python scripts/analysis/pmf_convergence.py --meta meta \
    --frames_per_ns <your actual output frequency>

# Stage 3: structural analysis
python scripts/analysis/anisotropy.py --dcd tot.dcd \
    --pdb reference_all_atom.pdb --ref reference_all_atom.pdb \
    --tica AA_tica_model.pkl --contacts contact_pairs.txt \
    --colorbar colorbar_hex_colors.txt

python scripts/analysis/nonnative_contacts.py --dcd tot.dcd \
    --pdb reference_all_atom.pdb --ref reference_all_atom.pdb \
    --npy unfold_shape_analysis_results.npy

# Stage 3: Gillespie kinetics (trap-involved vs. trap-blocked)
python scripts/analysis/gillespie_kinetics.py \
    --pmf_involved PMF_2d.out --pmf_blocked PMF_2d_masked.out \
    --diffusion_pkl drift_diffusion_wham_grid.pkl \
    --n_trajectories 1000 --connectivity 8-neighbor \
    --output_prefix gillespie
```

## Notes on reproducibility and data limitations

- Filenames used as script defaults (e.g. `reference_all_atom.pdb`,
  `protein_upside.pdb`) are generic placeholders; the exact input files
  used in this study are archived on Zenodo (see link above).
- Residue numbering: alignment/feature selections in these scripts use
  local trajectory topology numbering (e.g. residues 68-92), which
  corresponds to residues 445-469 in the native pertactin sequence
  numbering (PDB 1DAB) reported in the manuscript Methods.
- The position-dependent diffusion map used in the Gillespie kinetics
  analysis has incomplete direct coverage in some regions of tIC space
  (particularly the extended/unfolded region), reflecting limited REUS/
  REAP sampling there rather than a code artifact; see manuscript Methods
  and Limitations for details on how undersampled grid cells were
  handled.

## Citation

If you use this code, please cite the manuscript: [citation to be added
upon publication].
