#!/usr/bin/env python3
"""
Align Upside coarse-grained (CG) trajectories to a reference structure.

Alignment is performed on residues 68-92 in this trajectory's local
topology numbering, which corresponds to residues 445-469 in the native
pertactin sequence numbering (PDB 1DAB) used in the paper's Methods
section. 

Usage
-----
    python 01_align_cg_trajectories.py <input.dcd> <output.dcd> [--topology protein_upside.pdb]

Parameters
----------
input.dcd : str
    Path to the raw Upside CG trajectory to align.
output.dcd : str
    Path where the aligned trajectory will be written.
--topology : str, optional
    Topology/reference PDB file (default: protein_upside.pdb). This file is used
    both as the trajectory topology and as the alignment reference.
"""

import argparse
import mdtraj as md


# Local topology numbering; equivalent to residues 445-469 in the native
# pertactin numbering (PDB 1DAB) reported in the paper's Methods section.
ALIGN_SELECTION = "resid 68 to 92"


def align_trajectory(input_dcd, output_dcd, topology_pdb):
    traj = md.load(input_dcd, top=topology_pdb)
    reference = md.load(topology_pdb)

    atom_selection = traj.topology.select(ALIGN_SELECTION)
    traj_aligned = traj.superpose(
        reference,
        atom_indices=atom_selection,
        ref_atom_indices=atom_selection,
    )

    traj_aligned.save(output_dcd)
    print(f"Aligned trajectory saved to: {output_dcd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dcd", help="Path to raw CG trajectory (.dcd)")
    parser.add_argument("output_dcd", help="Path to write aligned trajectory (.dcd)")
    parser.add_argument(
        "--topology",
        default="protein_upside.pdb",
        help="Topology/reference PDB file (default: protein_upside.pdb)",
    )
    args = parser.parse_args()

    align_trajectory(args.input_dcd, args.output_dcd, args.topology)
