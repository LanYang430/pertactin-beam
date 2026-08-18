#!/usr/bin/env python3
"""
Train the final CG (Upside) TICA model used throughout this study.

Parameters below (lag = 100 Upside steps = 4 ns, dim = 2, backbone
Cartesian coordinates over residues 0-91) were selected based on the
scan performed in 02_lagtime_scan.py.

Usage
-----
    python 03_train_cg_tica.py --pdb protein_upside.pdb --traj_glob "cg_aligned/*.dcd" \
        --out cg_tica_model.pkl

Output
------
    A pickled PyEMMA TICA model (default: cg_tica_model.pkl), used
    downstream by scripts/reap/project_aa_to_cg_tica.py to project
    all-atom trajectories into the CG-learned collective variable space.
"""

import argparse
import glob
import pickle

import numpy as np
import pyemma


RESIDUE_RANGE = range(0, 92)  # backbone atoms used as TICA features
LAG_TIME = 100  # Upside steps (= 4 ns)
N_DIM = 2


def train_cg_tica(pdb, traj_glob, out_path, lag=LAG_TIME, dim=N_DIM):
    files = sorted(glob.glob(traj_glob))
    if not files:
        raise FileNotFoundError(f"No trajectory files matched: {traj_glob}")

    reader = pyemma.coordinates.source(files, top=pdb)
    atom_indices = reader.featurizer.topology.select(
        "resid " + " ".join(map(str, RESIDUE_RANGE)) + " and backbone"
    )
    reader.featurizer.add_selection(atom_indices)
    print("Feature dimension:", reader.featurizer.dimension())

    tica = pyemma.coordinates.tica(reader, lag=lag, dim=dim)
    tica_output = tica.get_output()
    print("TICA output dimensions:", tica.dimension())
    print("Total frames:", sum(len(t) for t in tica_output))

    with open(out_path, "wb") as f:
        pickle.dump(tica, f)
    print(f"Model saved to: {out_path}")

    return tica, tica_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", default="protein_upside.pdb", help="CG topology PDB")
    parser.add_argument(
        "--traj_glob",
        default="cg_aligned/*.dcd",
        help="Glob pattern matching CG trajectory files",
    )
    parser.add_argument(
        "--out", default="cg_tica_model.pkl", help="Output path for the trained TICA model"
    )
    parser.add_argument("--lag", type=int, default=LAG_TIME, help="TICA lag time (frames)")
    parser.add_argument("--dim", type=int, default=N_DIM, help="Number of TICA dimensions")
    args = parser.parse_args()

    train_cg_tica(args.pdb, args.traj_glob, args.out, lag=args.lag, dim=args.dim)
