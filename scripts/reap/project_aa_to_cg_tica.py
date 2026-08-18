#!/usr/bin/env python3
"""
Project an all-atom (AA) trajectory into the CG-learned TICA collective
variable space, and save the result as a REAP-compatible input.npy file.

This is the bridge between the CG (Upside) TICA model trained in
scripts/cg_tica/03_train_cg_tica.py and the REAP adaptive sampling
workflow: each round of AA simulations is projected into CG tIC space
using this script, and REAP selects the next round's starting
structures from that projection.

Usage
-----
    python project_aa_to_cg_tica.py <aa_traj.dcd> <aa_topology.pdb> \
        --reference reference_all_atom.pdb --model cg_tica_model.pkl \
        --out input.npy
"""

import argparse
import pickle

import numpy as np
import mdtraj as md

np.float = float  # compatibility shim for older pickled TICA models

ALIGN_SELECTION = "protein and backbone and resid 68 to 92"
FEATURE_SELECTION = "resid 0 to 91 and (name CA or name N or name C)"


def load_and_featurize_aa(dcd_path, topology_pdb, reference_pdb):
    """
    Load an AA trajectory, align it to a reference structure, and extract
    backbone (N, CA, C) Cartesian coordinates for residues 0-91.

    Returns a flattened feature matrix of shape (n_frames, n_atoms * 3),
    matching the feature representation used to train the CG TICA model.
    """
    traj = md.load(dcd_path, top=topology_pdb)
    reference = md.load(reference_pdb)

    align_atoms = traj.topology.select(ALIGN_SELECTION)
    traj_aligned = traj.superpose(
        reference, atom_indices=align_atoms, ref_atom_indices=align_atoms
    )

    feature_atoms = traj_aligned.topology.select(FEATURE_SELECTION)
    xyz = traj_aligned.atom_slice(feature_atoms).xyz

    n_frames, n_atoms, _ = xyz.shape
    return xyz.reshape(n_frames, n_atoms * 3)


def project_to_cg_tica(dcd_path, topology_pdb, reference_pdb, model_path, out_path):
    with open(model_path, "rb") as f:
        tica = pickle.load(f)

    features = load_and_featurize_aa(dcd_path, topology_pdb, reference_pdb)
    projected = tica.transform(features)

    print("Projected CV shape:", projected.shape)

    # REAP requires C-contiguous arrays
    projected = np.ascontiguousarray(projected)

    np.save(out_path, projected)
    print(f"Saved: {out_path}")

    return projected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aa_dcd", help="Path to AA trajectory (.dcd)")
    parser.add_argument("aa_topology", help="Topology PDB for the AA trajectory")
    parser.add_argument(
        "--reference",
        default="reference_all_atom.pdb",
        help="Reference structure for alignment (default: reference_all_atom.pdb)",
    )
    parser.add_argument(
        "--model",
        default="cg_tica_model.pkl",
        help="Path to the CG-trained TICA model (default: cg_tica_model.pkl)",
    )
    parser.add_argument("--out", default="input.npy", help="Output path (default: input.npy)")
    args = parser.parse_args()

    project_to_cg_tica(args.aa_dcd, args.aa_topology, args.reference, args.model, args.out)
