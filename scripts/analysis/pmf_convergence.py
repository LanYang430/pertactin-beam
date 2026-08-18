#!/usr/bin/env python3
"""
REUS/WHAM PMF convergence analysis.

Checks whether the 2D PMF from WHAM has converged as a function of
per-window simulation time, by re-running WHAM on truncated versions of
each window's trajectory and comparing the resulting PMF to the PMF
computed from the full trajectories.

Workflow
--------
1. Run WHAM once on the full (untruncated) trajectories to get a
   reference PMF (PMF_final.out).
2. For each requested per-window simulation time, truncate every
   window's trajectory to that many frames, re-run WHAM, and save the
   resulting PMF.
3. Compute the RMSD between each truncated-time PMF and the reference
   PMF (only over grid points that are valid/sampled in both PMFs, and
   after anchoring each PMF to its own minimum).
4. Plot RMSD vs. simulation time. A curve that flattens out below the
   convergence threshold indicates the PMF has converged.

Requires the WHAM binary (wham-2d) to be available on PATH.

Usage
-----
    python pmf_convergence.py --meta meta --frames_per_ns 100 \
        --wham_params "Px=0 -1.5 3.5 100 Py=0 -3.0 3.5 130 0.00001 310 0"

--frames_per_ns has NO default and must be supplied explicitly: it must
match the actual coordinate-output frequency of your REUS windows (check
the number of lines in any one window's trajectory file). Guessing this
value silently mislabels the time axis even if the RMSD values themselves
are correct.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

WHAM_UNSAMPLED = 9999999.0


@contextmanager
def temporary_wham_files(meta_file, max_lines):
    """
    Create a temporary meta file pointing to truncated copies of each
    window's trajectory (first `max_lines` lines only), and clean up
    afterward.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="wham_temp_")

        with open(meta_file, "r") as f:
            meta_lines = f.readlines()

        temp_meta = os.path.join(temp_dir, "temp_meta")

        with open(temp_meta, "w") as meta_out:
            for line in meta_lines:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue

                original_traj = parts[0]
                window_params = parts[1:]

                base_name = os.path.basename(original_traj)
                temp_traj = os.path.join(temp_dir, base_name)

                try:
                    with open(original_traj, "r") as traj_in:
                        lines = traj_in.readlines()[:max_lines]

                    with open(temp_traj, "w") as traj_out:
                        traj_out.writelines(lines)

                    meta_out.write(f"{temp_traj} {' '.join(window_params)}\n")

                except FileNotFoundError:
                    print(f"Warning: {original_traj} not found, skipping...")
                    continue

        yield temp_meta

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def run_wham_and_save(meta_file, max_lines, wham_params, output_file):
    """Run WHAM on trajectories truncated to `max_lines`, saving the PMF."""
    with temporary_wham_files(meta_file, max_lines) as temp_meta:
        try:
            cmd = f"wham-2d {wham_params} {temp_meta} {output_file} 1"
            print(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

            data = np.loadtxt(output_file)
            return data[:, 0], data[:, 1], data[:, 2]  # x, y, pmf

        except subprocess.CalledProcessError as e:
            print(f"WHAM failed for {max_lines} lines: {e}")
            print(f"stderr: {e.stderr}")
            return None, None, None
        except Exception as e:
            print(f"Error processing WHAM output: {e}")
            return None, None, None


def calculate_rmsd(pmf1, pmf2):
    """
    RMSD between two PMFs, restricted to grid points valid in both, after
    anchoring each PMF to its own minimum. WHAM marks unsampled grid
    points with WHAM_UNSAMPLED (9999999.0).
    """
    valid_mask1 = np.isfinite(pmf1) & (np.abs(pmf1 - WHAM_UNSAMPLED) > 1e-3)
    valid_mask2 = np.isfinite(pmf2) & (np.abs(pmf2 - WHAM_UNSAMPLED) > 1e-3)
    valid_mask = valid_mask1 & valid_mask2

    valid_count = np.sum(valid_mask)
    print(f"  Total grid points: {len(pmf1)}")
    print(f"  PMF1 valid points: {np.sum(valid_mask1)}")
    print(f"  PMF2 valid points: {np.sum(valid_mask2)}")
    print(f"  Common valid points: {valid_count}")

    if valid_count == 0:
        print("  Warning: no common valid grid points!")
        return np.inf

    if valid_count < 10:
        print(f"  Warning: very few common valid points ({valid_count})")

    pmf1_valid = pmf1[valid_mask]
    pmf2_valid = pmf2[valid_mask]

    print(f"  PMF1 valid range: {pmf1_valid.min():.2f} - {pmf1_valid.max():.2f}")
    print(f"  PMF2 valid range: {pmf2_valid.min():.2f} - {pmf2_valid.max():.2f}")

    pmf1_anchored = pmf1_valid - np.min(pmf1_valid)
    pmf2_anchored = pmf2_valid - np.min(pmf2_valid)

    rmsd = np.sqrt(np.mean((pmf1_anchored - pmf2_anchored) ** 2))
    print(f"  RMSD = {rmsd:.6f} kcal/mol")

    return rmsd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", default="meta", help="WHAM meta file listing window trajectories and parameters")
    parser.add_argument(
        "--wham_params",
        default="Px=0 -1.5 3.5 100 Py=0 -3.0 3.5 130 0.00001 310 0",
        help="Parameters passed to the wham-2d command (grid range, bins, tolerance, temperature, etc.)",
    )
    parser.add_argument(
        "--frames_per_ns",
        type=float,
        required=True,
        help="Actual coordinate-output frequency of your REUS windows, in frames per ns. "
        "No default - must reflect the real trajectory output stride. Check the number of "
        "lines in one window's trajectory file to confirm.",
    )
    parser.add_argument(
        "--time_points_ns",
        type=float,
        nargs="+",
        default=list(np.arange(0.1, 2.3, 0.1)),
        help="Per-window simulation times (ns) at which to evaluate convergence.",
    )
    parser.add_argument("--convergence_threshold", type=float, default=0.5, help="RMSD threshold (kcal/mol) for calling the PMF converged.")
    parser.add_argument("--output_dir", default="convergence_analysis")

    return parser.parse_args()


def main():
    args = parse_args()

    time_points_ns = list(args.time_points_ns)
    time_points = [int(round(t * args.frames_per_ns)) for t in time_points_ns]

    print(f"Time points: {time_points_ns} ns")
    print(f"Corresponding frame counts: {time_points}")
    print(f"Assumed output frequency: one frame every {1000.0 / args.frames_per_ns:.1f} ps")

    work_dir = args.output_dir
    os.makedirs(work_dir, exist_ok=True)

    pmf_history_dir = os.path.join(work_dir, "pmf_history")
    os.makedirs(pmf_history_dir, exist_ok=True)

    rmsd_values = []
    time_ns = []

    print("=" * 60)
    print("Starting convergence analysis (saving all PMF history files)...")
    print("=" * 60)

    print("\nComputing reference PMF (full trajectories)...")
    final_pmf_file = os.path.join(work_dir, "PMF_final.out")
    cmd = f"wham-2d {args.wham_params} {args.meta} {final_pmf_file} 1"

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        data_final = np.loadtxt(final_pmf_file)
        pmf_final = data_final[:, 2]
        print(f"Reference PMF saved: {final_pmf_file}")
        print(f"  PMF grid points: {len(pmf_final)}")

        valid_final = np.sum(np.isfinite(pmf_final) & (np.abs(pmf_final - WHAM_UNSAMPLED) > 1e-3))
        print(f"  Valid grid points: {valid_final}")

    except Exception as e:
        print(f"Could not compute reference PMF: {e}")
        return

    print("\n" + "=" * 60)
    print("Analyzing PMF at each time point...")
    print("=" * 60)

    for i, max_lines in enumerate(time_points):
        current_time = time_points_ns[i]
        print(f"\n[{i + 1}/{len(time_points)}] Processing time point: {current_time:.1f} ns ({max_lines} frames)")

        pmf_file = os.path.join(pmf_history_dir, f"PMF_{current_time:.1f}ns.out")

        _, _, pmf_temp = run_wham_and_save(args.meta, max_lines, args.wham_params, pmf_file)

        if pmf_temp is not None:
            print(f"  PMF saved: {pmf_file}")
            rmsd = calculate_rmsd(pmf_temp, pmf_final)

            if np.isfinite(rmsd):
                rmsd_values.append(rmsd)
                time_ns.append(current_time)
                print(f"  RMSD vs final = {rmsd:.4f} kcal/mol")
            else:
                print("  RMSD invalid, skipping this time point")
        else:
            print("  WHAM failed, skipping this time point")

    if len(rmsd_values) == 0:
        print("\nNo valid RMSD data. Check:")
        print("  1. Whether --meta points to the correct file")
        print("  2. Whether the trajectory files exist")
        print("  3. Whether --frames_per_ns is correct")
        print("  4. Whether --wham_params is correct")
        return

    print("\n" + "=" * 60)
    print("Generating convergence plot...")
    print("=" * 60)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        time_ns, rmsd_values, "b-o", linewidth=2, markersize=8,
        markerfacecolor="white", markeredgewidth=2,
    )

    ax.set_xlabel("Simulation time per window (ns)", fontsize=20, fontweight="bold")
    ax.set_ylabel("RMSD from final PMF (kcal/mol)", fontsize=20, fontweight="bold")
    ax.set_title("REUS PMF Convergence Analysis", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(axis="both", labelsize=18)

    ax.axhline(
        y=args.convergence_threshold, color="r", linestyle="--", linewidth=4, alpha=0.5,
        label=f"Convergence threshold ({args.convergence_threshold} kcal/mol)",
    )
    ax.legend(fontsize=16)

    plt.tight_layout()

    png_file = os.path.join(work_dir, "rmsd_convergence.png")
    pdf_file = os.path.join(work_dir, "rmsd_convergence.pdf")
    plt.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_file, bbox_inches="tight")

    print("Saved plots:")
    print(f"  - {png_file}")
    print(f"  - {pdf_file}")

    data_file = os.path.join(work_dir, "rmsd_data.txt")
    np.savetxt(
        data_file,
        np.column_stack([time_ns, rmsd_values]),
        header="Time(ns) RMSD(kcal/mol)",
        fmt="%.6f",
        comments="",
    )
    print(f"Saved data: {data_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"Results in: {work_dir}/")
    print(f"PMF history files: {pmf_history_dir}/ ({len(time_points)} files)")
    print("\nConvergence summary:")
    print(f"  Initial RMSD (@ {time_ns[0]:.1f} ns): {rmsd_values[0]:.4f} kcal/mol")
    print(f"  Final RMSD (@ {time_ns[-1]:.1f} ns): {rmsd_values[-1]:.4f} kcal/mol")

    if rmsd_values[-1] < args.convergence_threshold:
        print(f"  PMF appears converged (RMSD < {args.convergence_threshold} kcal/mol)")
    elif rmsd_values[-1] < 2 * args.convergence_threshold:
        print(f"  PMF is roughly converged ({args.convergence_threshold} < RMSD < {2 * args.convergence_threshold} kcal/mol)")
    else:
        print(f"  PMF may not be fully converged (RMSD > {2 * args.convergence_threshold} kcal/mol)")


if __name__ == "__main__":
    main()
