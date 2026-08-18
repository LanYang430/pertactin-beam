#!/usr/bin/env python3
"""
Lag-time scan for TICA on the CG (Upside) trajectory.

This script reproduces the parameter-selection analysis used to justify
the final TICA lag time (lag = 100 Upside steps = 4 ns) applied in
03_train_cg_tica.py. The lag time was selected based on qualitative
convergence of the IC1/IC2 density landscape across a range of lag
times: [10, 50, 100, 1000, 2000, 5000] Upside steps, corresponding to
[0.4, 2, 4, 40, 80, 200] ns at 40 ps/step. It is provided for
transparency/reproducibility of the parameter choice, not as a step
that needs to be re-run to reproduce the paper's main results.

Usage
-----
    python 02_lagtime_scan.py --pdb protein_upside.pdb --traj_glob "cg_aligned/*.dcd"

Output
------
    tica_lagtime_scan.png - IC1 vs IC2 density at each scanned lag time
"""

import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import pyemma


RESIDUE_RANGE = range(0, 92)  # backbone atoms used as TICA features
LAG_TIMES = [10, 50, 100, 1000, 2000, 5000]  # Upside steps (0.4-200 ns)


def build_position_reader(pdb, traj_glob):
    files = sorted(glob.glob(traj_glob))
    if not files:
        raise FileNotFoundError(f"No trajectory files matched: {traj_glob}")

    reader = pyemma.coordinates.source(files, top=pdb)

    atom_indices = reader.featurizer.topology.select(
        "resid " + " ".join(map(str, RESIDUE_RANGE)) + " and backbone"
    )
    reader.featurizer.add_selection(atom_indices)
    print("Selected backbone atom positions dimension:", reader.featurizer.dimension())

    return reader


def plot_density_across_lagtimes(reader, lag_times, save_path="tica_lagtime_scan.png"):
    n_rows = len(lag_times) // 2 + len(lag_times) % 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for i, lag in enumerate(lag_times):
        tica = pyemma.coordinates.tica(reader, lag=lag)
        tica_output = tica.get_output()
        tica_concatenated = np.concatenate(tica_output)

        pyemma.plots.plot_density(
            tica_concatenated[:, 0], tica_concatenated[:, 1], ax=axes[i], logscale=True
        )
        axes[i].set_xlabel(f"IC 1 (lag={lag})")
        axes[i].set_ylabel("IC 2")

    for j in range(len(lag_times), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb", default="protein_upside.pdb", help="CG topology/reference PDB"
    )
    parser.add_argument(
        "--traj_glob",
        default="cg_aligned/*.dcd",
        help="Glob pattern matching aligned CG trajectory files",
    )
    args = parser.parse_args()

    reader = build_position_reader(args.pdb, args.traj_glob)
    plot_density_across_lagtimes(reader, lag_times=LAG_TIMES)
