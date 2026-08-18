#!/usr/bin/env python3
"""
Non-native contact count by state (A/B/C) within the simulated pertactin
segment.

This script reuses the AA-trained tICA coordinates (tIC1/tIC2) saved by
anisotropy.py. It does not use the unfolded-residue assignments from that
analysis, and it does not train or apply any tICA model itself.

Workflow:
1. Load tIC1/tIC2 from the saved analysis npy file.
2. Load the AA trajectory and reference structure, and align it with
   the same selection used in anisotropy.py.
3. Define native contacts within the simulated segment from the reference
   structure (CA-CA distance <= native_cutoff, sequence separation >=
   min_seq_sep); all other sufficiently separated CA pairs in the segment
   are treated as non-native candidate contacts.
4. For each frame, count how many non-native candidate pairs are within
   contact_cutoff of each other.
5. Assign each frame to state A, B, or C based on Euclidean distance to
   fixed centers in tIC1/tIC2 space.
6. Plot the non-native contact count distribution per state.

Usage
-----
    python nonnative_contacts.py --dcd tot.dcd --pdb reference_all_atom.pdb \
        --ref reference_all_atom.pdb --npy unfold_shape_analysis_results.npy

    # Replot from previously saved metrics without re-reading the DCD:
    python nonnative_contacts.py --replot_metrics non_native_contacts_metrics.npy
"""

import argparse
import os
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt


# ============================================================
# Plot style
# ============================================================
def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4


# ============================================================
# Load AA-trained tICA coordinates from anisotropy.py output
# ============================================================
def load_existing_tica_data(npy_file):
    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"Cannot find npy file: {npy_file}")

    data = np.load(npy_file, allow_pickle=True).item()

    required_keys = ["tica1", "tica2"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in {npy_file}: {key}")

    tica1 = np.asarray(data["tica1"], dtype=float)
    tica2 = np.asarray(data["tica2"], dtype=float)

    print(f"Loaded existing AA-tICA data from: {npy_file}")
    print(f"Frames in npy: {len(tica1)}")

    return tica1, tica2


# ============================================================
# Atom selection helpers
# ============================================================
def select_ca_by_resseq(traj, res_start, res_end):
    """
    Select CA atoms by PDB residue number, atom.residue.resSeq.
    This avoids ambiguity between MDTraj resid and PDB resSeq.
    """
    atom_indices = []
    resseqs = []

    for atom in traj.topology.atoms:
        if atom.name == "CA" and atom.residue.is_protein:
            resseq = atom.residue.resSeq
            if res_start <= resseq <= res_end:
                atom_indices.append(atom.index)
                resseqs.append(resseq)

    atom_indices = np.asarray(atom_indices, dtype=int)
    resseqs = np.asarray(resseqs, dtype=int)

    order = np.argsort(resseqs)
    atom_indices = atom_indices[order]
    resseqs = resseqs[order]

    if len(atom_indices) == 0:
        raise ValueError(f"No CA atoms found for resSeq {res_start}-{res_end}")

    return atom_indices, resseqs


# ============================================================
# Native / non-native contact definitions
# ============================================================
def build_contact_pairs_from_reference(
    reference,
    traj,
    res_start=377,
    res_end=482,
    native_cutoff=0.8,
    min_seq_sep=4,
):
    """
    Define native Ca-Ca contacts from the reference structure.

    Native contacts:
        residue pairs separated by >= min_seq_sep
        and Ca-Ca distance <= native_cutoff in the reference.

    Non-native candidate contacts:
        residue pairs separated by >= min_seq_sep
        and not native in the reference.
    """
    ref_ca_indices, ref_resseqs = select_ca_by_resseq(reference, res_start, res_end)
    traj_ca_indices, traj_resseqs = select_ca_by_resseq(traj, res_start, res_end)

    if len(ref_resseqs) != len(traj_resseqs):
        raise ValueError(
            "Reference and trajectory segment have different number of CA atoms."
        )

    if not np.all(ref_resseqs == traj_resseqs):
        raise ValueError(
            "Reference and trajectory segment residue numbers do not match."
        )

    n_res = len(ref_resseqs)

    candidate_local_pairs = []

    for i in range(n_res):
        for j in range(i + 1, n_res):
            if abs(ref_resseqs[j] - ref_resseqs[i]) >= min_seq_sep:
                candidate_local_pairs.append((i, j))

    candidate_local_pairs = np.asarray(candidate_local_pairs, dtype=int)

    ref_xyz = reference.xyz[0, ref_ca_indices, :]

    ref_dists = np.linalg.norm(
        ref_xyz[candidate_local_pairs[:, 0]] - ref_xyz[candidate_local_pairs[:, 1]],
        axis=1,
    )

    native_mask = ref_dists <= native_cutoff
    non_native_local_pairs = candidate_local_pairs[~native_mask]
    non_native_atom_pairs = traj_ca_indices[non_native_local_pairs]

    print("\n=== Contact definition ===")
    print(f"Residue range: {res_start}-{res_end}")
    print(f"Number of CA atoms: {n_res}")
    print(f"Candidate nonlocal residue pairs: {len(candidate_local_pairs)}")
    print(f"Native contacts in reference: {int(np.sum(native_mask))}")
    print(f"Non-native candidate contacts: {len(non_native_atom_pairs)}")
    print(f"Native cutoff: {native_cutoff} nm")

    return {
        "non_native_atom_pairs": non_native_atom_pairs,
    }


# ============================================================
# Non-native contact count
# ============================================================
def compute_non_native_contacts(
    traj,
    reference,
    res_start=377,
    res_end=482,
    native_cutoff=0.8,
    contact_cutoff=0.8,
    min_seq_sep=4,
    chunk_size=1000,
):
    contact_info = build_contact_pairs_from_reference(
        reference=reference,
        traj=traj,
        res_start=res_start,
        res_end=res_end,
        native_cutoff=native_cutoff,
        min_seq_sep=min_seq_sep,
    )

    non_native_atom_pairs = contact_info["non_native_atom_pairs"]

    n_frames = traj.n_frames
    non_native_contact_count = np.full(n_frames, np.nan)

    print("\n=== Computing non-native contact counts ===")
    print(f"Total frames: {n_frames}")
    print(f"Chunk size: {chunk_size}")

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        print(f"Processing frames {start} - {end - 1}")

        subtraj = traj[start:end]

        if len(non_native_atom_pairs) > 0:
            non_native_distances = md.compute_distances(
                subtraj,
                non_native_atom_pairs,
                periodic=False,
            )
            non_native_contact_count[start:end] = np.sum(
                non_native_distances <= contact_cutoff, axis=1
            )

    return {
        "non_native_contact_count": non_native_contact_count,
        "native_cutoff": native_cutoff,
        "contact_cutoff": contact_cutoff,
        "min_seq_sep": min_seq_sep,
    }


# ============================================================
# State assignment
# ============================================================
def assign_states_by_tic_distance(
    tica1,
    tica2,
    center_A=(-0.95, 0.00),
    center_B=(1.30, 1.35),
    center_C=(1.15, -1.25),
    radius=0.35,
):
    centers = {
        1: np.asarray(center_A, dtype=float),
        2: np.asarray(center_B, dtype=float),
        3: np.asarray(center_C, dtype=float),
    }

    xy = np.column_stack([tica1, tica2])
    state_id = np.zeros(len(tica1), dtype=int)

    for i, point in enumerate(xy):
        dists = {
            sid: np.linalg.norm(point - center)
            for sid, center in centers.items()
        }

        nearest_sid = min(dists, key=dists.get)

        if dists[nearest_sid] <= radius:
            state_id[i] = nearest_sid

    print("\n=== State assignment ===")
    print(f"A center: {center_A}")
    print(f"B center: {center_B}")
    print(f"C center: {center_C}")
    print(f"Radius: {radius}")

    for sid, name in zip([1, 2, 3], ["A", "B", "C"]):
        print(f"State {name}: {np.sum(state_id == sid)} frames")

    print(f"Unassigned: {np.sum(state_id == 0)} frames")

    return state_id


# ============================================================
# Save outputs
# ============================================================
def save_metrics(metrics, state_id, output_prefix):
    out_npy = f"{output_prefix}_metrics.npy"

    save_dict = dict(metrics)
    save_dict["state_id"] = state_id

    np.save(out_npy, save_dict)
    print(f"\nSaved metrics npy: {out_npy}")

    rows = np.column_stack([
        np.arange(len(metrics["tica1"])),
        metrics["tica1"],
        metrics["tica2"],
        metrics["non_native_contact_count"],
        state_id,
    ])

    out_txt = f"{output_prefix}_per_frame_metrics.txt"

    np.savetxt(
        out_txt,
        rows,
        header=(
            "frame AA_tIC1 AA_tIC2 "
            "non_native_contact_count "
            "state_id_0none_1A_2B_3C"
        ),
        fmt=["%d", "%.8f", "%.8f", "%.0f", "%d"],
    )

    print(f"Saved per-frame txt: {out_txt}")


# ============================================================
# Plot distribution
# ============================================================
def plot_state_distribution(
    values,
    state_id,
    ylabel,
    output_prefix,
    metric_name,
    ylimit=None,
    max_scatter_per_state=800,
):
    set_plot_style()

    labels = ["A", "B", "C"]
    data = []

    for sid in [1, 2, 3]:
        vals = values[(state_id == sid) & np.isfinite(values)]
        data.append(vals)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    parts = ax.violinplot(
        data,
        positions=[1, 2, 3],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.55)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)

    ax.boxplot(
        data,
        positions=[1, 2, 3],
        widths=0.22,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
    )

    rng = np.random.default_rng(1)

    for pos, vals in zip([1, 2, 3], data):
        if len(vals) == 0:
            continue

        if len(vals) > max_scatter_per_state:
            idx = rng.choice(len(vals), size=max_scatter_per_state, replace=False)
            vals_plot = vals[idx]
        else:
            vals_plot = vals

        jitter = rng.normal(loc=0.0, scale=0.035, size=len(vals_plot))
        ax.scatter(
            np.full(len(vals_plot), pos) + jitter,
            vals_plot,
            s=4,
            alpha=0.18,
            linewidths=0,
            color="black",
        )

    ax.set_title("Non-native contacts", fontsize=18, pad=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=16, width=1.0, length=4, direction="out")
    ax.tick_params(axis="x", width=1.6, length=4, direction="out")

    if ylimit is not None:
        ax.set_ylim(*ylimit)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png = f"{output_prefix}_{metric_name}_states.png"
    pdf = f"{output_prefix}_{metric_name}_states.pdf"

    plt.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved plot: {png}")
    print(f"Saved plot: {pdf}")

    print(f"\n=== {metric_name} by state ===")
    for label, vals in zip(labels, data):
        if len(vals) == 0:
            print(f"State {label}: no data")
        else:
            print(
                f"State {label}: n={len(vals)}, "
                f"mean={np.mean(vals):.4f}, "
                f"median={np.median(vals):.4f}, "
                f"std={np.std(vals):.4f}"
            )


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute non-native contact counts for States A/B/C."
    )

    parser.add_argument("--dcd", default="tot.dcd")
    parser.add_argument("--pdb", default="reference_all_atom.pdb")
    parser.add_argument("--ref", default="reference_all_atom.pdb")
    parser.add_argument("--npy", default="unfold_shape_analysis_results.npy")

    parser.add_argument("--res_start", type=int, default=377)
    parser.add_argument("--res_end", type=int, default=482)

    parser.add_argument("--native_cutoff", type=float, default=0.8)
    parser.add_argument("--contact_cutoff", type=float, default=0.8)
    parser.add_argument("--min_seq_sep", type=int, default=4)

    parser.add_argument("--chunk_size", type=int, default=1000)

    parser.add_argument(
        "--align_selection",
        default="protein and backbone and resid 68 to 92",
        help="Use same alignment selection as anisotropy.py.",
    )

    parser.add_argument("--no_align", action="store_true")

    parser.add_argument("--A", type=float, nargs=2, default=[-0.95, 0.00])
    parser.add_argument("--B", type=float, nargs=2, default=[1.30, 1.35])
    parser.add_argument("--C", type=float, nargs=2, default=[1.15, -1.25])
    parser.add_argument("--state_radius", type=float, default=0.35)

    parser.add_argument("--output_prefix", default="non_native_contacts")

    parser.add_argument(
        "--replot_metrics",
        default=None,
        help="Replot from existing metrics npy without rereading DCD.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_plot_style()

    if args.replot_metrics is not None:
        print(f"Replotting from existing metrics: {args.replot_metrics}")
        metrics = np.load(args.replot_metrics, allow_pickle=True).item()

        if "state_id" in metrics:
            state_id = metrics["state_id"]
        else:
            state_id = assign_states_by_tic_distance(
                metrics["tica1"],
                metrics["tica2"],
                center_A=tuple(args.A),
                center_B=tuple(args.B),
                center_C=tuple(args.C),
                radius=args.state_radius,
            )

        plot_state_distribution(
            values=metrics["non_native_contact_count"],
            state_id=state_id,
            ylabel="Non-native contact count",
            output_prefix=args.output_prefix,
            metric_name="non_native_contact_count",
        )
        return

    for f in [args.dcd, args.pdb, args.ref, args.npy]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Cannot find file: {f}")

    tica1, tica2 = load_existing_tica_data(args.npy)

    print("\nLoading trajectory...")
    traj = md.load(args.dcd, top=args.pdb)
    reference = md.load(args.ref)

    print(f"Trajectory frames: {traj.n_frames}")
    print(f"Trajectory atoms: {traj.n_atoms}")

    if traj.n_frames != len(tica1):
        raise ValueError(
            f"Frame mismatch: DCD has {traj.n_frames} frames, npy has {len(tica1)} frames."
        )

    if not args.no_align:
        print("\nAligning trajectory using same selection as anisotropy.py...")
        print(f"Alignment selection: {args.align_selection}")

        atom_indices = traj.topology.select(args.align_selection)
        ref_atom_indices = reference.topology.select(args.align_selection)

        print(f"Alignment atoms in traj: {len(atom_indices)}")
        print(f"Alignment atoms in ref: {len(ref_atom_indices)}")

        if len(atom_indices) == 0 or len(ref_atom_indices) == 0:
            print("Warning: alignment selection returned 0 atoms. Skipping alignment.")
        else:
            traj.superpose(
                reference,
                atom_indices=atom_indices,
                ref_atom_indices=ref_atom_indices,
            )

    contact_metrics = compute_non_native_contacts(
        traj=traj,
        reference=reference,
        res_start=args.res_start,
        res_end=args.res_end,
        native_cutoff=args.native_cutoff,
        contact_cutoff=args.contact_cutoff,
        min_seq_sep=args.min_seq_sep,
        chunk_size=args.chunk_size,
    )

    metrics = {
        "tica1": tica1,
        "tica2": tica2,
        **contact_metrics,
    }

    state_id = assign_states_by_tic_distance(
        tica1,
        tica2,
        center_A=tuple(args.A),
        center_B=tuple(args.B),
        center_C=tuple(args.C),
        radius=args.state_radius,
    )

    save_metrics(metrics, state_id, args.output_prefix)

    with open(f"{args.output_prefix}_state_definition.txt", "w") as f:
        f.write("State assignment by Euclidean distance in tIC1/tIC2 space\n")
        f.write(f"A center: {args.A}\n")
        f.write(f"B center: {args.B}\n")
        f.write(f"C center: {args.C}\n")
        f.write(f"radius: {args.state_radius}\n")
        f.write("state_id: 0=None, 1=A, 2=B, 3=C\n")

    plot_state_distribution(
        values=metrics["non_native_contact_count"],
        state_id=state_id,
        ylabel="Non-native contact count",
        output_prefix=args.output_prefix,
        metric_name="non_native_contact_count",
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
