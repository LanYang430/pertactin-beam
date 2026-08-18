#!/usr/bin/env python3
"""
Anisotropy of the unfolded region, mapped onto the AA-trained tICA CV space.

Workflow:
1. Load an AA trajectory and align it (residues 68-92 in local topology
   numbering; residues 445-469 in the native pertactin/PDB 1DAB
   numbering used in the paper's Methods section).
2.Project backbone coordinates onto the pre-trained all-atom TICA model
   used for the PMF and kinetic analyses. 
3. Identify, per frame, which residues are "unfolded" based on a set of
   monitored native-contact pairs (a contact is considered broken if the
   CA-CA distance exceeds a cutoff).
4. Compute shape descriptors (anisotropy, principal axis ratios) from
   the coordinates of the unfolded residues in each frame.
5. Grid-average anisotropy over the 2D TICA space and plot it as a
   heatmap.

Usage
-----
    python anisotropy.py --dcd tot.dcd --pdb reference_all_atom.pdb \
        --ref reference_all_atom.pdb --tica AA_tica_model.pkl \
        --contacts contact_pairs.txt --colorbar colorbar_hex_colors.txt

Contact pairs file format (one pair per line, native/PDB residue numbering):
   379 397
   380 398
   381 399
"""

import mdtraj as md
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


ALIGNMENT_SELECTION = "protein and backbone and resid 68 to 92"


def parse_contact_pairs(contact_file):
    """
    Parse a contact-pair file.

    File format: one contact pair per line, as "resid1 resid2" or
    "resid1-resid2".
    """
    contact_pairs = []

    if not os.path.exists(contact_file):
        print(f"Error: contact pairs file not found: {contact_file}")
        return contact_pairs

    with open(contact_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                if '-' in line:
                    parts = line.split('-')
                else:
                    parts = line.split()

                if len(parts) == 2:
                    res1, res2 = int(parts[0]), int(parts[1])
                    contact_pairs.append((res1, res2))
                else:
                    print(f"Warning: malformed line {line_num}, skipping: {line}")
            except ValueError:
                print(f"Warning: could not parse line {line_num} as integers, skipping: {line}")

    print(f"Loaded {len(contact_pairs)} contact pairs")
    return contact_pairs


def identify_unfolded_residues(traj_ca, contact_pairs, cutoff_distance=0.8):
    """
    Identify, for each frame, which residues are in an "unfolded" state.

    Parameters
    ----------
    traj_ca : mdtraj.Trajectory
        Trajectory containing only CA atoms.
    contact_pairs : list of (int, int)
        Native-contact CA pairs to monitor.
    cutoff_distance : float
        Distance cutoff (nm) above which a contact is considered broken.

    Returns
    -------
    unfolded_residues_per_frame : list of list of int
    contact_status_per_frame : list of dict
    """
    print(f"Identifying unfolded residues, distance cutoff: {cutoff_distance:.2f} nm")

    ca_residues = np.array([atom.residue.resSeq for atom in traj_ca.topology.atoms])
    print(f"CA residue range: {min(ca_residues)} - {max(ca_residues)}")

    valid_pairs = []
    pair_indices = []

    for res1, res2 in contact_pairs:
        idx1_list = np.where(ca_residues == res1)[0]
        idx2_list = np.where(ca_residues == res2)[0]

        if len(idx1_list) == 0:
            print(f"Warning: CA atom for residue {res1} not found")
            continue
        if len(idx2_list) == 0:
            print(f"Warning: CA atom for residue {res2} not found")
            continue

        idx1, idx2 = idx1_list[0], idx2_list[0]
        valid_pairs.append((res1, res2))
        pair_indices.append((idx1, idx2))

    print(f"Valid contact pairs: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("Error: no valid contact pairs")
        return None, None

    unfolded_residues_per_frame = []
    contact_status_per_frame = []

    for frame_idx in range(traj_ca.n_frames):
        coords = traj_ca.xyz[frame_idx]

        broken_contacts = []
        frame_contact_status = {}

        for (res1, res2), (idx1, idx2) in zip(valid_pairs, pair_indices):
            dist = np.linalg.norm(coords[idx1] - coords[idx2])
            is_broken = dist > cutoff_distance
            frame_contact_status[f'{res1}-{res2}'] = {
                'distance': dist,
                'is_broken': is_broken
            }

            if is_broken:
                broken_contacts.extend([res1, res2])

        unfolded_residues = sorted(set(broken_contacts))

        unfolded_residues_per_frame.append(unfolded_residues)
        contact_status_per_frame.append(frame_contact_status)

        if (frame_idx + 1) % 1000 == 0:
            print(f"Processed {frame_idx + 1}/{traj_ca.n_frames} frames")

    return unfolded_residues_per_frame, contact_status_per_frame


def calculate_shape_descriptors(coordinates):
    """
    Compute shape descriptors (anisotropy, principal axis ratios) from a
    set of 3D coordinates.
    """
    if coordinates is None or len(coordinates) < 3:
        return None

    centroid = np.mean(coordinates, axis=0)
    centered_coords = coordinates - centroid

    cov_matrix = np.cov(centered_coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda1, lambda2, lambda3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

    descriptors = {}

    lambda_mean = (lambda1 + lambda2 + lambda3) / 3
    if lambda_mean > 0:
        descriptors['anisotropy'] = (
            (lambda1 - lambda_mean) ** 2
            + (lambda2 - lambda_mean) ** 2
            + (lambda3 - lambda_mean) ** 2
        ) / (2 * lambda_mean ** 2)
    else:
        descriptors['anisotropy'] = 0

    descriptors['axis_ratio_12'] = lambda1 / lambda2 if lambda2 > 0 else float('inf')
    if lambda3 > 0:
        descriptors['axis_ratio_13'] = lambda1 / lambda3
        descriptors['axis_ratio_23'] = lambda2 / lambda3
    else:
        descriptors['axis_ratio_13'] = float('inf')
        descriptors['axis_ratio_23'] = float('inf')

    descriptors['eigenvalues'] = (lambda1, lambda2, lambda3)

    return descriptors


def load_and_process_trajectory(dcd_file, pdb_file, reference_pdb, tica_model_file,
                                 contact_pairs_file, cutoff_distance=0.8):
    """
    Load a trajectory, project it onto a pre-trained TICA model, identify
    unfolded residues per frame, and compute shape descriptors for the
    unfolded region.
    """
    print(f"Loading trajectory: {dcd_file}")
    print(f"Topology: {pdb_file}")
    print(f"Reference: {reference_pdb}")
    print(f"Contact pairs: {contact_pairs_file}")

    try:
        traj = md.load(dcd_file, top=pdb_file)
        reference = md.load(reference_pdb)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return None, None, None

    print(f"Loaded trajectory with {traj.n_frames} frames")

    try:
        atom_indices = traj.topology.select(ALIGNMENT_SELECTION)
        ref_atom_indices = reference.topology.select(ALIGNMENT_SELECTION)

        print(f"Alignment atoms: {len(atom_indices)}")

        traj_aligned = traj.superpose(
            reference, atom_indices=atom_indices, ref_atom_indices=ref_atom_indices
        )
    except Exception as e:
        print(f"Error during alignment: {e}")
        return None, None, None

    protein_atoms = traj_aligned.topology.select("protein")
    traj_protein_only = traj_aligned.atom_slice(protein_atoms)

    print(f"Protein atoms: {len(protein_atoms)}")

    try:
        with open(tica_model_file, 'rb') as f:
            tica_model = pickle.load(f)
        print(f"Loaded TICA model from {tica_model_file}")
    except Exception as e:
        print(f"Error loading TICA model: {e}")
        return None, None, None

    backbone_indices = traj_protein_only.topology.select("name N or name CA or name C")

    print(f"Selected {len(backbone_indices)} backbone atoms")
    print(f"Expected features: {len(backbone_indices) * 3}")

    traj_backbone = traj_protein_only.atom_slice(backbone_indices)

    xyz_coords = traj_backbone.xyz
    n_frames, n_atoms, _ = xyz_coords.shape
    xyz_flattened = xyz_coords.reshape(n_frames, n_atoms * 3)

    try:
        tica_result = tica_model.transform(xyz_flattened)
        print(f"TICA result shape: {tica_result.shape}")
    except Exception as e:
        print(f"Error applying TICA: {e}")
        return None, None, None

    contact_pairs = parse_contact_pairs(contact_pairs_file)
    if not contact_pairs:
        return None, None, None

    ca_indices = traj_protein_only.topology.select("name CA")
    traj_ca = traj_protein_only.atom_slice(ca_indices)

    print(f"Identifying unfolded residues and computing shape descriptors "
          f"for {len(ca_indices)} CA atoms...")

    unfolded_residues_per_frame, contact_status_per_frame = identify_unfolded_residues(
        traj_ca, contact_pairs, cutoff_distance)

    if unfolded_residues_per_frame is None:
        return None, None, None

    ca_residues = np.array([atom.residue.resSeq for atom in traj_ca.topology.atoms])

    anisotropy_data = []
    axis_ratio_12_data = []
    axis_ratio_13_data = []
    unfold_stats = []

    total_monitored_residues = set()
    for res1, res2 in contact_pairs:
        total_monitored_residues.add(res1)
        total_monitored_residues.add(res2)

    for frame_idx in range(traj_ca.n_frames):
        unfolded_residues = unfolded_residues_per_frame[frame_idx]

        unfold_fraction = (
            len(unfolded_residues) / len(total_monitored_residues)
            if total_monitored_residues else 0
        )

        unfold_stats.append({
            'frame': frame_idx,
            'unfolded_residues': unfolded_residues,
            'unfold_count': len(unfolded_residues),
            'unfold_fraction': unfold_fraction,
            'contact_status': contact_status_per_frame[frame_idx]
        })

        if len(unfolded_residues) < 3:
            # Too few unfolded residues to compute a shape descriptor
            anisotropy_data.append(np.nan)
            axis_ratio_12_data.append(np.nan)
            axis_ratio_13_data.append(np.nan)
            continue

        unfolded_ca_indices = []
        for res_id in unfolded_residues:
            idx_list = np.where(ca_residues == res_id)[0]
            if len(idx_list) > 0:
                unfolded_ca_indices.append(idx_list[0])

        if len(unfolded_ca_indices) < 3:
            anisotropy_data.append(np.nan)
            axis_ratio_12_data.append(np.nan)
            axis_ratio_13_data.append(np.nan)
            continue

        coords = traj_ca.xyz[frame_idx][unfolded_ca_indices]
        descriptors = calculate_shape_descriptors(coords)

        if descriptors is not None:
            anisotropy_data.append(descriptors['anisotropy'])
            axis_ratio_12_data.append(descriptors['axis_ratio_12'])
            axis_ratio_13_data.append(descriptors['axis_ratio_13'])
        else:
            anisotropy_data.append(np.nan)
            axis_ratio_12_data.append(np.nan)
            axis_ratio_13_data.append(np.nan)

        if (frame_idx + 1) % 1000 == 0:
            print(f"Processed {frame_idx + 1}/{traj_ca.n_frames} frames")

    shape_data = {
        'anisotropy': anisotropy_data,
        'axis_ratio_12': axis_ratio_12_data,
        'axis_ratio_13': axis_ratio_13_data
    }

    return tica_result, shape_data, unfold_stats


def plot_shape_analysis(tica1, tica2, shape_data, unfold_stats,
                         grid_size=0.05, min_points=5,
                         output_prefix='unfold_shape_tica',
                         colorbar_file='colorbar_hex_colors.txt'):
    """
    Plot grid-averaged anisotropy of the unfolded region over the 2D
    TICA space.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['axes.linewidth'] = 1.4
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5

    with open(colorbar_file, 'r') as f:
        colors = [line.strip() for line in f.readlines() if line.strip()]

    def adjust_colormap(colors, brightness=1.2, saturation=0.9):
        rgb_colors = np.array([mcolors.hex2color(c) for c in colors])
        hsv_colors = rgb_to_hsv(rgb_colors.reshape(-1, 1, 3)).reshape(-1, 3)
        hsv_colors[:, 1] *= saturation
        hsv_colors[:, 2] *= brightness
        hsv_colors = np.clip(hsv_colors, 0, 1)
        rgb_adjusted = hsv_to_rgb(hsv_colors.reshape(-1, 1, 3)).reshape(-1, 3)
        return mcolors.ListedColormap(rgb_adjusted)

    if colorbar_file and os.path.exists(colorbar_file):
        with open(colorbar_file, 'r') as f:
	    colors = [line.strip() for line in f.readlines() if line.strip()]
        custom_cmap = adjust_colormap(colors, brightness=1.15, saturation=0.95)
    else:
        print(f"Warning: colorbar file not found: {colorbar_file}; using default colormap.")
        custom_cmap = "viridis"

    anisotropy_values = np.array(shape_data['anisotropy'])
    anisotropy_values[np.isinf(anisotropy_values)] = np.nan

    print(f"\nProcessing shape analysis:")
    print(f"Total frames: {len(anisotropy_values)}")
    print(f"Valid anisotropy values: {np.sum(~np.isnan(anisotropy_values))}")

    if np.sum(~np.isnan(anisotropy_values)) == 0:
        print("No valid anisotropy values")
        return

    valid_mask = ~(np.isnan(tica1) | np.isnan(tica2) | np.isnan(anisotropy_values))
    tica1_valid = tica1[valid_mask]
    tica2_valid = tica2[valid_mask]
    anisotropy_valid = anisotropy_values[valid_mask]

    if len(tica1_valid) == 0:
        print("No valid data to plot")
        return

    tica1_min, tica1_max = np.min(tica1_valid), np.max(tica1_valid)
    tica2_min, tica2_max = np.min(tica2_valid), np.max(tica2_valid)

    tica1_edges = np.arange(tica1_min - grid_size / 2, tica1_max + grid_size, grid_size)
    tica2_edges = np.arange(tica2_min - grid_size / 2, tica2_max + grid_size, grid_size)

    tica1_indices = np.digitize(tica1_valid, tica1_edges) - 1
    tica2_indices = np.digitize(tica2_valid, tica2_edges) - 1

    n_grid1 = len(tica1_edges) - 1
    n_grid2 = len(tica2_edges) - 1

    grid_sums = np.zeros((n_grid2, n_grid1))
    grid_counts = np.zeros((n_grid2, n_grid1))

    for i in range(len(tica1_valid)):
        i1, i2 = tica1_indices[i], tica2_indices[i]
        if 0 <= i1 < n_grid1 and 0 <= i2 < n_grid2:
            grid_sums[i2, i1] += anisotropy_valid[i]
            grid_counts[i2, i1] += 1

    grid_averages = np.full((n_grid2, n_grid1), np.nan)
    mask = grid_counts >= min_points
    grid_averages[mask] = grid_sums[mask] / grid_counts[mask]

    X, Y = np.meshgrid(tica1_edges, tica2_edges)

    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(X, Y, grid_averages, shading='auto', cmap=custom_cmap)

    cbar = plt.colorbar(c, ax=ax, pad=0.03)
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label('Anisotropy', fontsize=38, rotation=90, labelpad=15)

    ax.set_xlabel('AA-trained tIC1', fontsize=34)
    ax.set_ylabel('AA-trained tIC2', fontsize=34)
    # Axis limits matching the manuscript figure's data range
    ax.set_xlim(-1.18, 3.12)
    ax.set_ylim(-2.83, 2.92)
    ax.tick_params(axis='both', labelsize=28)

    plt.tight_layout()

    output_file = f"{output_prefix}_anisotropy_grid{grid_size}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")

    plt.show()

    print(f"\n=== Anisotropy Statistics ===")
    valid_ani_vals = anisotropy_values[~np.isnan(anisotropy_values)]

    if len(valid_ani_vals) > 0:
        print(f"Anisotropy range: {np.min(valid_ani_vals):.4f} - {np.max(valid_ani_vals):.4f}")
        print(f"Mean anisotropy: {np.mean(valid_ani_vals):.4f}")
        print(f"Median anisotropy: {np.median(valid_ani_vals):.4f}")
        print(f"Valid grid cells: {np.sum(~np.isnan(grid_averages))}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python anisotropy.py [options]")
        print("Options:")
        print("  --dcd <file>         DCD trajectory file (default: tot.dcd)")
        print("  --pdb <file>         PDB topology file (default: reference_all_atom.pdb)")
        print("  --ref <file>         Reference PDB file (default: reference_all_atom.pdb)")
        print("  --tica <file>        TICA model file (default: AA_tica_model.pkl)")
        print("  --contacts <file>    Contact pairs file (default: contact_pairs.txt)")
        print("  --colorbar <file>    Custom colorbar file (default: colorbar_hex_colors.txt)")
        print("  --cutoff <distance>  Contact distance cutoff in nm (default: 0.8)")
        print("  --grid <size>        Grid size for averaging (default: 0.05)")
        print("  --min_points <n>     Minimum points per grid (default: 5)")
        print("")
        print("Contact pairs file format (one pair per line):")
        print("  10 25")
        print("  15 30")
        print("  20-35")
        return

    dcd_file = "tot.dcd"
    pdb_file = "reference_all_atom.pdb"
    reference_pdb = "reference_all_atom.pdb"
    tica_model_file = "AA_tica_model.pkl"
    contact_pairs_file = "contact_pairs.txt"
    colorbar_file = "colorbar_hex_colors.txt"
    cutoff_distance = 0.8  # nm
    grid_size = 0.05
    min_points = 5

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--dcd' and i + 1 < len(sys.argv):
            dcd_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--pdb' and i + 1 < len(sys.argv):
            pdb_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--ref' and i + 1 < len(sys.argv):
            reference_pdb = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--tica' and i + 1 < len(sys.argv):
            tica_model_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--contacts' and i + 1 < len(sys.argv):
            contact_pairs_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--colorbar' and i + 1 < len(sys.argv):
            colorbar_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--cutoff' and i + 1 < len(sys.argv):
            cutoff_distance = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--grid' and i + 1 < len(sys.argv):
            grid_size = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--min_points' and i + 1 < len(sys.argv):
            min_points = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    print("=== Unfold-based Shape Analysis with TICA ===")
    print(f"DCD file: {dcd_file}")
    print(f"PDB file: {pdb_file}")
    print(f"Reference: {reference_pdb}")
    print(f"TICA model: {tica_model_file}")
    print(f"Contact pairs file: {contact_pairs_file}")
    print(f"Contact cutoff: {cutoff_distance} nm")
    print(f"Colorbar file: {colorbar_file}")
    print(f"Grid size: {grid_size}")
    print(f"Min points per grid: {min_points}")

    print("\nAnalysis workflow:")
    print("1. Load trajectory and calculate TICA coordinates")
    print("2. Identify unfolded residues based on broken contacts")
    print("3. Calculate shape descriptors (anisotropy, axis ratios) for unfolded regions")
    print("4. Plot results in TICA space")

    required_files = [dcd_file, pdb_file, reference_pdb, tica_model_file, contact_pairs_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: file not found: {file_path}")
            return

    tica_result, shape_data, unfold_stats = load_and_process_trajectory(
        dcd_file, pdb_file, reference_pdb, tica_model_file, contact_pairs_file, cutoff_distance)

    if tica_result is None or shape_data is None or unfold_stats is None:
        print("Failed to process trajectory")
        return

    tica1 = tica_result[:, 0]
    tica2 = tica_result[:, 1]

    print(f"\nTICA coordinate ranges:")
    print(f"TICA 1: {np.min(tica1):.3f} to {np.max(tica1):.3f}")
    print(f"TICA 2: {np.min(tica2):.3f} to {np.max(tica2):.3f}")

    print("\nGenerating shape analysis plots...")
    plot_shape_analysis(tica1, tica2, shape_data, unfold_stats,
                         grid_size, min_points,
                         'unfold_shape_tica', colorbar_file)

    output_data = {
        'tica1': tica1,
        'tica2': tica2,
        'shape_data': shape_data,
        'unfold_stats': unfold_stats,
        'cutoff_distance': cutoff_distance
    }
    np.save('unfold_shape_analysis_results.npy', output_data)
    print("\nResults saved to: unfold_shape_analysis_results.npy")

    unfold_counts = [stats['unfold_count'] for stats in unfold_stats]
    unfold_fractions = [stats['unfold_fraction'] for stats in unfold_stats]

    anisotropy_values = np.array(shape_data['anisotropy'])
    valid_anisotropy = anisotropy_values[~np.isnan(anisotropy_values)]

    axis_ratio_12_values = np.array(shape_data['axis_ratio_12'])
    valid_axis_ratio = axis_ratio_12_values[~np.isnan(axis_ratio_12_values) & (axis_ratio_12_values < 100)]

    print(f"\n=== Summary Statistics ===")
    print(f"Total frames analyzed: {len(unfold_stats)}")
    print(f"Frames with valid anisotropy: {len(valid_anisotropy)}")
    print(f"Frames with valid axis ratio: {len(valid_axis_ratio)}")
    print(f"Unfold count range: {np.min(unfold_counts)} - {np.max(unfold_counts)} residues")
    print(f"Mean unfold count: {np.mean(unfold_counts):.1f} residues")
    print(f"Unfold fraction range: {np.min(unfold_fractions):.3f} - {np.max(unfold_fractions):.3f}")
    print(f"Mean unfold fraction: {np.mean(unfold_fractions):.3f}")

    if len(valid_anisotropy) > 0:
        print(f"\nAnisotropy statistics:")
        print(f"  Range: {np.min(valid_anisotropy):.4f} - {np.max(valid_anisotropy):.4f}")
        print(f"  Mean: {np.mean(valid_anisotropy):.4f}")
        print(f"  Median: {np.median(valid_anisotropy):.4f}")
        print(f"  Std: {np.std(valid_anisotropy):.4f}")

    if len(valid_axis_ratio) > 0:
        print(f"\nAxis ratio (lambda1/lambda2) statistics:")
        print(f"  Range: {np.min(valid_axis_ratio):.3f} - {np.max(valid_axis_ratio):.3f}")
        print(f"  Mean: {np.mean(valid_axis_ratio):.3f}")
        print(f"  Median: {np.median(valid_axis_ratio):.3f}")
        print(f"  Std: {np.std(valid_axis_ratio):.3f}")

    print(f"\n=== Contact Pair Analysis ===")
    if len(unfold_stats) > 0:
        first_frame = unfold_stats[0]
        contact_names = list(first_frame['contact_status'].keys())
        print(f"Monitored contact pairs: {len(contact_names)}")

        for contact_name in contact_names[:10]:
            distances = []
            broken_count = 0

            for stats in unfold_stats:
                contact_info = stats['contact_status'][contact_name]
                distances.append(contact_info['distance'])
                if contact_info['is_broken']:
                    broken_count += 1

            broken_fraction = broken_count / len(unfold_stats)
            avg_distance = np.mean(distances)

            print(f"  {contact_name}: mean distance {avg_distance:.3f} nm, "
                  f"broken fraction {broken_fraction:.3f}")

        if len(contact_names) > 10:
            print(f"  ... {len(contact_names) - 10} more contact pairs")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
