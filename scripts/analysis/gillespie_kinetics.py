#!/usr/bin/env python3
"""
Gillespie / kinetic Monte Carlo folding simulation on a REUS-derived PMF.

This is the simple, paper-consistent version of the Gillespie workflow. It
builds a kinetic network on a 2D tIC1/tIC2 PMF grid and runs Gillespie
trajectories from an extended/unfolded region to the folded basin.

Key assumptions matching the manuscript Methods
-----------------------------------------------
1. The PMF is defined on an all-atom-trained tIC1/tIC2 grid.
2. PMF values are in kcal/mol by default.
3. Local diffusion coefficients D(tIC1,tIC2) are in tIC^2/ps.
4. Transition rates are computed as

       k_ij = D_ij / Delta_r_ij^2 * exp[-(F_j - F_i) / (2 k_B T)]

   where D_ij = (D_i + D_j) / 2, and Delta_r_ij is the distance between
   neighboring grid-cell centers in tIC space. For up/down/left/right
   neighbors, Delta_r_ij^2 = dx^2 (or dy^2). For diagonal 8-neighbor jumps,
   Delta_r_ij^2 = dx^2 + dy^2 (NOT just dx^2 - diagonal neighbors are
   farther apart than axis-aligned neighbors).
5. Gillespie waiting times are therefore in ps.
6. Output folding curves are saved in microseconds.

Diffusion map handling
-----------------------
If --diffusion_pkl is provided, D(tIC1,tIC2) is loaded from it. If the
stored grid is smaller than the PMF grid (e.g. D was only estimated on
grid cells with enough MSD samples), it is placed onto the full PMF grid
by matching tIC1/tIC2 bin coordinates stored in the pkl. Any PMF-valid
cell that still lacks a real D value after this placement is filled with
the median of the cells that DO have a real value - a single global
number, with no local-neighbor interpolation. This is intentionally the
simplest possible fill, to make it easy to isolate whether more elaborate
fill logic changes the resulting network shape.

If --diffusion_pkl is omitted, a uniform D (--uniform_D) is used for all
PMF-valid cells; this is for testing/sensitivity analysis only.

Example: single PMF
-------------------
    python gillespie_kinetics.py \
        --pmf PMF_2d.out \
        --diffusion_pkl drift_diffusion_wham_grid.pkl \
        --n_trajectories 1000 \
        --max_time_ps 1e30 \
        --connectivity 8-neighbor \
        --condition_name trap_involved

Example: two PMFs used in the original analysis
-------------------------------------------------
    python gillespie_kinetics.py \
        --pmf_involved PMF_2d.out \
        --pmf_blocked PMF_2d_masked.out \
        --diffusion_pkl drift_diffusion_wham_grid.pkl \
        --n_trajectories 1000 \
        --max_time_ps 1e30 \
        --connectivity 8-neighbor \
        --output_prefix gillespie

Optional trap-blocking mode
---------------------------
For a single PMF, State B can also be removed in code with --block_state_B.
The two-PMF mode does not apply --block_state_B because the blocked PMF is
assumed to be pre-masked already.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


# ---------------------------------------------------------------------
# Constants and units
# ---------------------------------------------------------------------
KB_KCAL_PER_MOL_K = 0.00198720425864083
KB_KJ_PER_MOL_K = 0.00831446261815324


@dataclass
class GridData:
    tica1_bins: np.ndarray          # shape (nx,)
    tica2_bins: np.ndarray          # shape (ny,)
    pmf_2d: np.ndarray              # shape (ny, nx), indexed as [j, i]
    valid_mask: np.ndarray          # shape (ny, nx), indexed as [j, i]
    diffusion_2d: np.ndarray        # shape (ny, nx), tIC^2/ps, indexed as [j, i]
    dx: float
    dy: float


# ---------------------------------------------------------------------
# Loading the PMF
# ---------------------------------------------------------------------
def load_wham_pmf(pmf_file: str, invalid_value: float = 9999999.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a WHAM-style PMF_2d.out file.

    Expected columns:
        tIC1  tIC2  free_energy

    Returns
    -------
    tica1_bins, tica2_bins, pmf_2d, valid_mask
        pmf_2d and valid_mask have shape (ny, nx), indexed as [j, i].
    """
    if not os.path.exists(pmf_file):
        raise FileNotFoundError(f"PMF file not found: {pmf_file}")

    data = np.genfromtxt(pmf_file, skip_header=1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("PMF file must contain at least three columns: tIC1 tIC2 PMF")

    x = data[:, 0]
    y = data[:, 1]
    free_energy = data[:, 2]

    # Use all grid coordinates to define the grid, not only valid PMF points.
    # This is important for pre-masked PMFs: an entire row/column could be
    # invalid, but the grid shape should still match the unmasked PMF and
    # the diffusion map.
    coordinate_mask = np.isfinite(x) & np.isfinite(y)
    valid_data = coordinate_mask & np.isfinite(free_energy) & (free_energy != invalid_value)

    tica1_bins = np.unique(x[coordinate_mask])
    tica2_bins = np.unique(y[coordinate_mask])

    nx = len(tica1_bins)
    ny = len(tica2_bins)

    pmf_2d = np.full((ny, nx), np.nan, dtype=float)

    x_index = {val: idx for idx, val in enumerate(tica1_bins)}
    y_index = {val: idx for idx, val in enumerate(tica2_bins)}

    for xv, yv, fv, ok in zip(x, y, free_energy, valid_data):
        if not ok:
            continue
        i = x_index[xv]
        j = y_index[yv]
        pmf_2d[j, i] = fv

    valid_mask = np.isfinite(pmf_2d)

    dx = float(np.median(np.diff(tica1_bins))) if nx > 1 else np.nan
    dy = float(np.median(np.diff(tica2_bins))) if ny > 1 else np.nan

    print("=== Loaded PMF grid ===")
    print(f"PMF file: {pmf_file}")
    print(f"tIC1 range: {tica1_bins.min():.3f} to {tica1_bins.max():.3f}; n={nx}; dx={dx:.6f}")
    print(f"tIC2 range: {tica2_bins.min():.3f} to {tica2_bins.max():.3f}; n={ny}; dy={dy:.6f}")
    print(f"Valid PMF grid points: {int(np.sum(valid_mask))}/{pmf_2d.size}")

    return tica1_bins, tica2_bins, pmf_2d, valid_mask


# ---------------------------------------------------------------------
# Loading and (minimally) filling the diffusion map
# ---------------------------------------------------------------------
def _first_existing_key(data: Dict, keys: List[str]) -> Optional[str]:
    """Return the first key present in a dictionary."""
    for key in keys:
        if key in data:
            return key
    return None


def _get_source_bins(data: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Try to find tIC1/tIC2 bin arrays stored in a diffusion pkl.

    Needed when the diffusion map was computed only on a smaller
    sampled/valid region and therefore has a shape smaller than the PMF
    grid, so it can be placed onto the full PMF grid by coordinate.
    """
    x_key = _first_existing_key(
        data,
        ["tica1_bins", "tic1_bins", "tIC1_bins", "x_bins", "xi",
         "tica1_centers", "tic1_centers", "x_centers", "grid_x"],
    )
    y_key = _first_existing_key(
        data,
        ["tica2_bins", "tic2_bins", "tIC2_bins", "y_bins", "yi",
         "tica2_centers", "tic2_centers", "y_centers", "grid_y"],
    )

    src_x = np.asarray(data[x_key], dtype=float) if x_key is not None else None
    src_y = np.asarray(data[y_key], dtype=float) if y_key is not None else None

    if src_x is not None and src_x.ndim == 2:
        src_x = np.unique(src_x[np.isfinite(src_x)])
    if src_y is not None and src_y.ndim == 2:
        src_y = np.unique(src_y[np.isfinite(src_y)])

    return src_x, src_y


def _map_partial_grid_by_bins(
    arr_yx: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    pmf_tica1_bins: np.ndarray,
    pmf_tica2_bins: np.ndarray,
    pmf_shape: Tuple[int, int],
) -> np.ndarray:
    """Place a smaller (ny_src, nx_src) diffusion array onto the full PMF grid by matching bin coordinates."""
    out = np.full(pmf_shape, np.nan, dtype=float)
    dx = float(np.median(np.diff(pmf_tica1_bins))) if len(pmf_tica1_bins) > 1 else np.inf
    dy = float(np.median(np.diff(pmf_tica2_bins))) if len(pmf_tica2_bins) > 1 else np.inf
    tol_x = max(abs(dx) * 0.51, 1e-8)
    tol_y = max(abs(dy) * 0.51, 1e-8)

    n_placed = 0
    for sj, yv in enumerate(src_y):
        if not np.isfinite(yv):
            continue
        tj = int(np.argmin(np.abs(pmf_tica2_bins - yv)))
        if abs(pmf_tica2_bins[tj] - yv) > tol_y:
            continue
        for si, xv in enumerate(src_x):
            if not np.isfinite(xv):
                continue
            ti = int(np.argmin(np.abs(pmf_tica1_bins - xv)))
            if abs(pmf_tica1_bins[ti] - xv) > tol_x:
                continue
            out[tj, ti] = arr_yx[sj, si]
            n_placed += 1

    print(f"Placed {n_placed}/{arr_yx.size} diffusion values onto the PMF grid by coordinate matching.")
    return out


def _place_on_pmf_grid(
    arr: np.ndarray,
    pmf_shape: Tuple[int, int],
    name: str,
    data: Dict,
    pmf_tica1_bins: np.ndarray,
    pmf_tica2_bins: np.ndarray,
) -> np.ndarray:
    """Return `arr` reshaped/placed to match pmf_shape (ny, nx).

    Tries, in order: exact shape match, transpose, and (if the pkl has
    stored tIC1/tIC2 bin coordinates) coordinate-based placement of a
    smaller grid onto the full PMF grid, leaving unplaced cells as NaN.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.shape == pmf_shape:
        return arr
    if arr.T.shape == pmf_shape:
        print(f"Transposing {name} from {arr.shape} to {pmf_shape} to match PMF indexing.")
        return arr.T

    print(f"WARNING: {name} has shape {arr.shape}, different from PMF shape {pmf_shape}.")
    src_x, src_y = _get_source_bins(data)
    if src_x is None or src_y is None:
        raise ValueError(
            f"{name} has shape {arr.shape}, cannot match PMF shape {pmf_shape}, "
            "and no tIC1/tIC2 bin coordinates were found in the pkl to place it."
        )

    if arr.shape == (len(src_y), len(src_x)):
        arr_yx = arr
    elif arr.shape == (len(src_x), len(src_y)):
        arr_yx = arr.T
    else:
        raise ValueError(
            f"{name} shape {arr.shape} does not match source bin dimensions "
            f"(ny={len(src_y)}, nx={len(src_x)}) in either orientation."
        )

    return _map_partial_grid_by_bins(arr_yx, src_x, src_y, pmf_tica1_bins, pmf_tica2_bins, pmf_shape)


def load_diffusion_map(
    diffusion_pkl: Optional[str],
    pmf_shape: Tuple[int, int],
    valid_mask: np.ndarray,
    pmf_tica1_bins: np.ndarray,
    pmf_tica2_bins: np.ndarray,
    uniform_D: float = 1e-11,
) -> np.ndarray:
    """Load position-dependent D(tIC1,tIC2), or use a uniform value.

    Minimal fill strategy: any PMF-valid cell without a positive, finite
    D value (whether because the pkl's grid didn't cover it, or the
    value itself was missing/invalid) is filled with the median of the
    cells that DO have a real value. No local-neighbor interpolation is
    done - this is the simplest possible fill.

    Accepted diffusion keys: diffusion_2d, diffusion, D, or
    diffusion_tica1 + diffusion_tica2 (averaged).
    """
    if diffusion_pkl is None:
        print("=== Diffusion map ===")
        print("WARNING: No --diffusion_pkl provided. Using uniform diffusion.")
        print("This is for testing/sensitivity analysis; manuscript analysis should use local D(tIC1,tIC2).")
        D = np.full(pmf_shape, uniform_D, dtype=float)
        D[~valid_mask] = np.nan
        print(f"Uniform D = {uniform_D:.6e} tIC^2/ps")
        return D

    if not os.path.exists(diffusion_pkl):
        raise FileNotFoundError(f"Diffusion pkl not found: {diffusion_pkl}")

    with open(diffusion_pkl, "rb") as f:
        data = pickle.load(f)

    print("=== Loaded diffusion map ===")
    print(f"Diffusion pkl: {diffusion_pkl}")

    if "diffusion_2d" in data:
        D = _place_on_pmf_grid(data["diffusion_2d"], pmf_shape, "diffusion_2d", data, pmf_tica1_bins, pmf_tica2_bins)
    elif "diffusion" in data:
        D = _place_on_pmf_grid(data["diffusion"], pmf_shape, "diffusion", data, pmf_tica1_bins, pmf_tica2_bins)
    elif "D" in data:
        D = _place_on_pmf_grid(data["D"], pmf_shape, "D", data, pmf_tica1_bins, pmf_tica2_bins)
    elif "diffusion_tica1" in data and "diffusion_tica2" in data:
        d1 = _place_on_pmf_grid(data["diffusion_tica1"], pmf_shape, "diffusion_tica1", data, pmf_tica1_bins, pmf_tica2_bins)
        d2 = _place_on_pmf_grid(data["diffusion_tica2"], pmf_shape, "diffusion_tica2", data, pmf_tica1_bins, pmf_tica2_bins)
        D = 0.5 * (d1 + d2)
        print("Using scalar D = 0.5 * (diffusion_tica1 + diffusion_tica2).")
    else:
        keys = ", ".join(sorted(data.keys()))
        raise KeyError(
            "Could not find a diffusion map in the pkl. Expected one of "
            f"'diffusion_2d', 'diffusion', 'D', or both 'diffusion_tica1' and 'diffusion_tica2'. "
            f"Available keys: {keys}"
        )

    D = np.asarray(D, dtype=float)
    D[~valid_mask] = np.nan
    D[~np.isfinite(D)] = np.nan
    D[D <= 0] = np.nan

    raw_valid = valid_mask & np.isfinite(D) & (D > 0)
    n_raw_valid = int(np.sum(raw_valid))
    n_pmf_valid = int(np.sum(valid_mask))
    print(f"Raw valid D cells: {n_raw_valid}/{n_pmf_valid} PMF-valid grid points")

    if n_raw_valid == 0:
        raise ValueError("No valid positive diffusion values found anywhere in the pkl for this PMF grid.")

    D_values = D[raw_valid]
    D_min = float(np.min(D_values))
    D_median = float(np.median(D_values))
    D_max = float(np.max(D_values))
    print(f"D over raw valid cells (tIC^2/ps): min={D_min:.3e}, median={D_median:.3e}, max={D_max:.3e}")

    # Minimal fill: any PMF-valid cell without a real D value gets the
    # overall median. No neighbor-based interpolation.
    missing = valid_mask & ~raw_valid
    n_missing = int(np.sum(missing))
    D[missing] = D_median
    print(f"Filled {n_missing} PMF-valid cells with the median D = {D_median:.3e} tIC^2/ps")

    D_all_valid = D[valid_mask]
    print(
        f"D over ALL PMF-valid cells after fill (tIC^2/ps): "
        f"min={np.min(D_all_valid):.3e}, median={np.median(D_all_valid):.3e}, max={np.max(D_all_valid):.3e}"
    )

    return D


def plot_diffusion_map(
    D: np.ndarray,
    tica1_bins: np.ndarray,
    tica2_bins: np.ndarray,
    output_prefix: str = "diffusion_map",
):
    """Save a simple diagnostic heatmap of the final D(tIC1,tIC2) used in the network."""
    extent = [tica1_bins[0], tica1_bins[-1], tica2_bins[0], tica2_bins[-1]]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, origin="lower", aspect="auto", cmap="plasma", extent=extent)
    ax.set_xlabel("tIC1")
    ax.set_ylabel("tIC2")
    ax.set_title("Diffusion D used in network (tIC^2/ps)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    out_png = f"{output_prefix}_diffusion_used_in_network.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot: {out_png}")


def load_grid_data(
    pmf_file: str,
    diffusion_pkl: Optional[str],
    invalid_value: float,
    uniform_D: float,
) -> GridData:
    tica1_bins, tica2_bins, pmf_2d, valid_mask = load_wham_pmf(pmf_file, invalid_value=invalid_value)
    diffusion_2d = load_diffusion_map(
        diffusion_pkl=diffusion_pkl,
        pmf_shape=pmf_2d.shape,
        valid_mask=valid_mask,
        pmf_tica1_bins=tica1_bins,
        pmf_tica2_bins=tica2_bins,
        uniform_D=uniform_D,
    )

    dx = float(np.median(np.diff(tica1_bins)))
    dy = float(np.median(np.diff(tica2_bins)))

    if not np.isclose(dx, dy, rtol=1e-3, atol=1e-8):
        print(f"NOTE: dx ({dx}) and dy ({dy}) differ. Diagonal distances use dx^2 + dy^2.")

    return GridData(
        tica1_bins=tica1_bins,
        tica2_bins=tica2_bins,
        pmf_2d=pmf_2d,
        valid_mask=valid_mask,
        diffusion_2d=diffusion_2d,
        dx=dx,
        dy=dy,
    )


# ---------------------------------------------------------------------
# State definitions and masking
# ---------------------------------------------------------------------
def in_rectangle(tic1: float, tic2: float, tica1_range: Iterable[float], tica2_range: Iterable[float]) -> bool:
    xmin, xmax = tica1_range
    ymin, ymax = tica2_range
    return xmin <= tic1 <= xmax and ymin <= tic2 <= ymax


def in_circle(tic1: float, tic2: float, center: Tuple[float, float], radius: float) -> bool:
    point = np.asarray([tic1, tic2], dtype=float)
    center_arr = np.asarray(center, dtype=float)
    return np.linalg.norm(point - center_arr) <= radius


def build_state_masks(
    grid: GridData,
    folded_tic1_range: Tuple[float, float],
    folded_tic2_range: Tuple[float, float],
    unfolded_tic1_range: Tuple[float, float],
    unfolded_tic2_range: Tuple[float, float],
    folded_pmf_max: Optional[float],
    unfolded_pmf_min: Optional[float],
    block_state_B: bool,
    state_B_center: Tuple[float, float],
    state_radius: float,
) -> Dict[str, np.ndarray]:
    """Build boolean masks for folded, unfolded, and blocked State B regions."""
    ny, nx = grid.pmf_2d.shape
    folded_mask = np.zeros((ny, nx), dtype=bool)
    unfolded_mask = np.zeros((ny, nx), dtype=bool)
    block_mask = np.zeros((ny, nx), dtype=bool)

    for j in range(ny):
        for i in range(nx):
            if not grid.valid_mask[j, i]:
                continue

            tic1 = grid.tica1_bins[i]
            tic2 = grid.tica2_bins[j]
            pmf = grid.pmf_2d[j, i]

            folded = in_rectangle(tic1, tic2, folded_tic1_range, folded_tic2_range)
            if folded_pmf_max is not None:
                folded = folded and pmf <= folded_pmf_max

            unfolded = in_rectangle(tic1, tic2, unfolded_tic1_range, unfolded_tic2_range)
            if unfolded_pmf_min is not None:
                unfolded = unfolded and pmf >= unfolded_pmf_min

            if folded:
                folded_mask[j, i] = True
            if unfolded:
                unfolded_mask[j, i] = True

            if block_state_B and in_circle(tic1, tic2, state_B_center, state_radius):
                block_mask[j, i] = True

    print("=== State masks ===")
    print(f"Folded grid cells: {int(np.sum(folded_mask))}")
    print(f"Unfolded/extended starting grid cells: {int(np.sum(unfolded_mask))}")
    if block_state_B:
        print(f"Blocked State B grid cells: {int(np.sum(block_mask))}")
        print(f"State B center: {state_B_center}; radius: {state_radius}")

    if np.sum(folded_mask) == 0:
        raise ValueError("No folded grid cells found. Check folded state definition.")
    if np.sum(unfolded_mask) == 0:
        raise ValueError("No unfolded starting grid cells found. Check unfolded state definition.")

    return {
        "folded_mask": folded_mask,
        "unfolded_mask": unfolded_mask,
        "block_mask": block_mask,
    }


# ---------------------------------------------------------------------
# Kinetic network
# ---------------------------------------------------------------------
def build_kinetic_network(
    grid: GridData,
    masks: Dict[str, np.ndarray],
    temperature_K: float = 310.0,
    pmf_units: str = "kcal",
    connectivity: str = "8-neighbor",
) -> Dict:
    """Build kinetic network using the paper rate formula.

    Rates have units ps^-1 because D has units tIC^2/ps and Delta_r has
    units tIC. For diagonal 8-neighbor jumps, Delta_r^2 = dx^2 + dy^2
    (not just dx^2), since diagonal neighbors are farther apart than
    axis-aligned neighbors.
    """
    if pmf_units == "kcal":
        kT = KB_KCAL_PER_MOL_K * temperature_K
        unit_label = "kcal/mol"
    elif pmf_units == "kJ":
        kT = KB_KJ_PER_MOL_K * temperature_K
        unit_label = "kJ/mol"
    else:
        raise ValueError("--pmf_units must be 'kcal' or 'kJ'")

    print("=== Building kinetic network ===")
    print(f"Temperature: {temperature_K:.2f} K")
    print(f"kT: {kT:.6f} {unit_label}")
    print(f"Connectivity: {connectivity}")

    if connectivity == "4-neighbor":
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == "8-neighbor":
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        raise ValueError("--connectivity must be '4-neighbor' or '8-neighbor'")

    accessible_mask = (
        grid.valid_mask
        & np.isfinite(grid.pmf_2d)
        & np.isfinite(grid.diffusion_2d)
        & (grid.diffusion_2d > 0)
        & ~masks["block_mask"]
    )

    ny, nx = grid.pmf_2d.shape
    node_map = np.full((ny, nx), -1, dtype=int)
    valid_js, valid_is = np.where(accessible_mask)

    for node_id, (j, i) in enumerate(zip(valid_js, valid_is)):
        node_map[j, i] = node_id

    n_nodes = len(valid_js)
    print(f"Accessible network nodes: {n_nodes}")

    folded_nodes = []
    unfolded_nodes = []
    for node_id, (j, i) in enumerate(zip(valid_js, valid_is)):
        if masks["folded_mask"][j, i]:
            folded_nodes.append(node_id)
        if masks["unfolded_mask"][j, i]:
            unfolded_nodes.append(node_id)

    print(f"Folded nodes in accessible network: {len(folded_nodes)}")
    print(f"Unfolded starting nodes in accessible network: {len(unfolded_nodes)}")
    if len(folded_nodes) == 0:
        raise ValueError("No folded nodes remain in accessible network.")
    if len(unfolded_nodes) == 0:
        raise ValueError("No unfolded starting nodes remain in accessible network.")

    edges = []
    rates = []

    dx2 = grid.dx ** 2
    dy2 = grid.dy ** 2

    for node_id, (j, i) in enumerate(zip(valid_js, valid_is)):
        F_i = grid.pmf_2d[j, i]
        D_i = grid.diffusion_2d[j, i]

        for dj, di in neighbor_offsets:
            nj = j + dj
            ni = i + di
            if nj < 0 or nj >= ny or ni < 0 or ni >= nx:
                continue

            neighbor_node = node_map[nj, ni]
            if neighbor_node < 0:
                continue

            # Distance^2 between this cell and the neighbor cell center.
            # Axis-aligned moves: only one of di, dj is nonzero.
            # Diagonal moves (8-neighbor connectivity): both are nonzero,
            # so the true distance is sqrt(dx^2 + dy^2), not just dx.
            if di != 0 and dj != 0:
                delta_r2 = dx2 + dy2
            else:
                delta_r2 = dx2 if di != 0 else dy2

            F_j = grid.pmf_2d[nj, ni]
            D_j = grid.diffusion_2d[nj, ni]
            D_ij = 0.5 * (D_i + D_j)
            delta_F = F_j - F_i

            # Paper-consistent discretized Smoluchowski rate:
            # k_ij = D_ij / |r_ij|^2 * exp[-(F_j - F_i)/(2 kBT)]
            k_ij = (D_ij / delta_r2) * np.exp(-delta_F / (2.0 * kT))

            if np.isfinite(k_ij) and k_ij > 0:
                edges.append((node_id, neighbor_node))
                rates.append(k_ij)

    adjacency = csr_matrix(
        (rates, ([e[0] for e in edges], [e[1] for e in edges])),
        shape=(n_nodes, n_nodes),
    )

    n_components, component_labels = connected_components(adjacency, directed=False)
    print(f"Network edges: {len(edges)}")
    print(f"Connected components: {n_components}")
    if n_components > 1:
        component_sizes = [int(np.sum(component_labels == c)) for c in range(n_components)]
        print(f"Connected component sizes: {component_sizes}")

    node_coordinates = np.column_stack([grid.tica1_bins[valid_is], grid.tica2_bins[valid_js]])

    network = {
        "n_nodes": n_nodes,
        "node_map": node_map,
        "valid_js": valid_js,
        "valid_is": valid_is,
        "node_coordinates": node_coordinates,
        "adjacency_matrix": adjacency,
        "edges": edges,
        "edge_rates_ps_inv": np.asarray(rates, dtype=float),
        "folded_nodes": folded_nodes,
        "unfolded_nodes": unfolded_nodes,
        "pmf_2d": grid.pmf_2d,
        "diffusion_2d": grid.diffusion_2d,
        "tica1_bins": grid.tica1_bins,
        "tica2_bins": grid.tica2_bins,
        "dx": grid.dx,
        "dy": grid.dy,
        "temperature_K": temperature_K,
        "kT": kT,
        "pmf_units": pmf_units,
        "connectivity": connectivity,
        "connected_components": n_components,
        "component_labels": component_labels,
    }

    return network


# ---------------------------------------------------------------------
# Gillespie simulation
# ---------------------------------------------------------------------
def gillespie_simulation(
    network: Dict,
    n_trajectories: int = 1000,
    max_time_ps: float = 1e30,
    seed: int = 1,
    track_trajectories: bool = False,
) -> Dict:
    """Run Gillespie folding trajectories.

    Starting nodes are sampled uniformly from the unfolded/extended region.
    A trajectory is successful once it reaches any folded node.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    adjacency = network["adjacency_matrix"].tocsr()
    n_nodes = network["n_nodes"]
    folded_nodes = set(network["folded_nodes"])
    unfolded_nodes = list(network["unfolded_nodes"])

    node_neighbors: List[np.ndarray] = []
    node_rates: List[np.ndarray] = []

    for node in range(n_nodes):
        row_start = adjacency.indptr[node]
        row_end = adjacency.indptr[node + 1]
        neighbors = adjacency.indices[row_start:row_end]
        rates = adjacency.data[row_start:row_end]
        node_neighbors.append(neighbors)
        node_rates.append(rates)

    folding_times_ps = []
    trajectory_data = [] if track_trajectories else None

    print("=== Running Gillespie simulations ===")
    print(f"Trajectories: {n_trajectories}")
    print(f"Max time: {max_time_ps:.3e} ps")
    print(f"Random seed: {seed}")

    for traj_id in tqdm(range(n_trajectories), desc="Gillespie trajectories"):
        current_node = rng.choice(unfolded_nodes)
        current_time = 0.0

        nodes = [current_node] if track_trajectories else None
        times = [current_time] if track_trajectories else None

        success = False
        fail_reason = None

        while current_time < max_time_ps:
            if current_node in folded_nodes:
                folding_times_ps.append(current_time)
                success = True
                break

            neighbors = node_neighbors[current_node]
            rates = node_rates[current_node]

            if len(neighbors) == 0:
                fail_reason = "dead_node"
                break

            total_rate = float(np.sum(rates))
            if total_rate <= 0 or not np.isfinite(total_rate):
                fail_reason = "zero_or_invalid_rate"
                break

            waiting_time = -np.log(np_rng.random()) / total_rate
            current_time += waiting_time

            probabilities = rates / total_rate
            current_node = int(np_rng.choice(neighbors, p=probabilities))

            if track_trajectories:
                nodes.append(current_node)
                times.append(current_time)

        if current_time >= max_time_ps and not success:
            fail_reason = "timeout"

        if track_trajectories:
            trajectory_data.append(
                {
                    "traj_id": traj_id,
                    "success": success,
                    "folding_time_ps": current_time if success else None,
                    "fail_reason": fail_reason,
                    "nodes": nodes,
                    "times_ps": times,
                    "coordinates": network["node_coordinates"][nodes].tolist(),
                }
            )

    folding_times_ps = np.asarray(folding_times_ps, dtype=float)
    success_rate = len(folding_times_ps) / n_trajectories * 100.0

    print("=== Gillespie results ===")
    print(f"Successful folding: {len(folding_times_ps)}/{n_trajectories} ({success_rate:.1f}%)")

    results = {
        "folding_times_ps": folding_times_ps,
        "n_trajectories": n_trajectories,
        "success_rate_percent": success_rate,
        "trajectory_data": trajectory_data,
        "network": network,
    }

    if len(folding_times_ps) > 0:
        mean_ps = float(np.mean(folding_times_ps))
        median_ps = float(np.median(folding_times_ps))
        std_ps = float(np.std(folding_times_ps))

        results.update(
            {
                "mean_folding_time_ps": mean_ps,
                "median_folding_time_ps": median_ps,
                "std_folding_time_ps": std_ps,
                "mean_folding_time_us": mean_ps / 1e6,
                "median_folding_time_us": median_ps / 1e6,
                "folding_rate_ps_inv": 1.0 / mean_ps,
                "folding_rate_s_inv": 1.0 / (mean_ps * 1e-12),
            }
        )

        print(f"Mean folding time:   {mean_ps:.3e} ps = {mean_ps / 1e6:.3f} us")
        print(f"Median folding time: {median_ps:.3e} ps = {median_ps / 1e6:.3f} us")
        print(f"Std folding time:    {std_ps:.3e} ps = {std_ps / 1e6:.3f} us")
        print(f"Mean folding rate:   {1.0 / mean_ps:.3e} ps^-1 = {1.0 / (mean_ps * 1e-12):.3e} s^-1")

    return results


# ---------------------------------------------------------------------
# Plotting and output
# ---------------------------------------------------------------------
def set_plot_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_folding_curve(results: Dict, output_prefix: str, condition_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Save cumulative folding percentage vs time in microseconds."""
    folding_times_ps = np.asarray(results["folding_times_ps"], dtype=float)
    n_trajectories = int(results["n_trajectories"])

    if len(folding_times_ps) == 0:
        print("No successful folding times; skipping folding curve output.")
        return np.array([]), np.array([])

    folding_times_sorted_ps = np.sort(folding_times_ps)
    time_us = folding_times_sorted_ps / 1e6
    folded_percent = np.arange(1, len(folding_times_sorted_ps) + 1) / n_trajectories * 100.0

    out_txt = f"{output_prefix}_{condition_name}_folding_curve.txt"
    np.savetxt(
        out_txt,
        np.column_stack([time_us, folded_percent]),
        header="Time_us\tFolded_percentage",
        fmt="%.8f\t%.4f",
    )
    print(f"Saved folding curve data: {out_txt}")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(time_us, folded_percent, linewidth=2)

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("% folded")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)

    median_us = np.median(folding_times_sorted_ps) / 1e6
    ax.axhline(50, linestyle=":", linewidth=1)
    ax.axvline(median_us, linestyle="--", linewidth=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png = f"{output_prefix}_{condition_name}_folding_curve.png"
    out_pdf = f"{output_prefix}_{condition_name}_folding_curve.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved folding curve plot: {out_png}")
    print(f"Saved folding curve plot: {out_pdf}")

    return time_us, folded_percent


def save_results(results: Dict, output_prefix: str, condition_name: str):
    out_pkl = f"{output_prefix}_{condition_name}_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full results: {out_pkl}")

    network = results["network"]
    summary_file = f"{output_prefix}_{condition_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"condition_name: {condition_name}\n")
        f.write(f"n_trajectories: {results['n_trajectories']}\n")
        f.write(f"success_rate_percent: {results['success_rate_percent']:.6f}\n")
        f.write(f"connectivity: {network['connectivity']}\n")
        f.write(f"temperature_K: {network['temperature_K']:.6f}\n")
        f.write(f"kT_{network['pmf_units']}_per_mol: {network['kT']:.8f}\n")
        f.write(f"dx_tIC: {network['dx']:.8f}\n")
        f.write(f"dy_tIC: {network['dy']:.8f}\n")
        f.write(f"n_nodes: {network['n_nodes']}\n")
        f.write(f"n_edges: {len(network['edges'])}\n")
        f.write(f"n_folded_nodes: {len(network['folded_nodes'])}\n")
        f.write(f"n_unfolded_nodes: {len(network['unfolded_nodes'])}\n")
        if len(results["folding_times_ps"]) > 0:
            f.write(f"mean_folding_time_ps: {results['mean_folding_time_ps']:.8e}\n")
            f.write(f"median_folding_time_ps: {results['median_folding_time_ps']:.8e}\n")
            f.write(f"mean_folding_time_us: {results['mean_folding_time_us']:.8e}\n")
            f.write(f"median_folding_time_us: {results['median_folding_time_us']:.8e}\n")
            f.write(f"folding_rate_s_inv: {results['folding_rate_s_inv']:.8e}\n")
    print(f"Saved summary: {summary_file}")


# ---------------------------------------------------------------------
# Running one or two conditions
# ---------------------------------------------------------------------
def run_condition(
    pmf_file: str,
    diffusion_pkl: Optional[str],
    invalid_value: float,
    uniform_D: float,
    folded_tic1_range: Tuple[float, float],
    folded_tic2_range: Tuple[float, float],
    unfolded_tic1_range: Tuple[float, float],
    unfolded_tic2_range: Tuple[float, float],
    folded_pmf_max: Optional[float],
    unfolded_pmf_min: Optional[float],
    temperature_K: float,
    pmf_units: str,
    connectivity: str,
    n_trajectories: int,
    max_time_ps: float,
    seed: int,
    track_trajectories: bool,
    output_prefix: str,
    condition_name: str,
    block_state_B: bool = False,
    state_B_center: Tuple[float, float] = (1.30, 1.35),
    state_radius: float = 0.35,
) -> Dict:
    """Run one Gillespie condition from one PMF file."""
    print("\n" + "=" * 72)
    print(f"Running condition: {condition_name}")
    print(f"PMF: {pmf_file}")
    print("=" * 72)

    grid = load_grid_data(
        pmf_file=pmf_file,
        diffusion_pkl=diffusion_pkl,
        invalid_value=invalid_value,
        uniform_D=uniform_D,
    )

    plot_diffusion_map(
        grid.diffusion_2d,
        grid.tica1_bins,
        grid.tica2_bins,
        output_prefix=f"{output_prefix}_{condition_name}",
    )

    masks = build_state_masks(
        grid=grid,
        folded_tic1_range=folded_tic1_range,
        folded_tic2_range=folded_tic2_range,
        unfolded_tic1_range=unfolded_tic1_range,
        unfolded_tic2_range=unfolded_tic2_range,
        folded_pmf_max=folded_pmf_max,
        unfolded_pmf_min=unfolded_pmf_min,
        block_state_B=block_state_B,
        state_B_center=state_B_center,
        state_radius=state_radius,
    )

    network = build_kinetic_network(
        grid=grid,
        masks=masks,
        temperature_K=temperature_K,
        pmf_units=pmf_units,
        connectivity=connectivity,
    )

    results = gillespie_simulation(
        network=network,
        n_trajectories=n_trajectories,
        max_time_ps=max_time_ps,
        seed=seed,
        track_trajectories=track_trajectories,
    )

    results["condition_name"] = condition_name
    results["pmf_file"] = pmf_file
    results["block_state_B_in_code"] = block_state_B

    time_us, folded_percent = save_folding_curve(
        results, output_prefix=output_prefix, condition_name=condition_name
    )
    results["folding_curve_time_us"] = time_us
    results["folding_curve_percent"] = folded_percent

    save_results(results, output_prefix=output_prefix, condition_name=condition_name)

    return results


def save_comparison_curve(
    results_list: List[Dict],
    output_prefix: str,
    xlim_reference_condition: str = "trap_blocked",
    xlim_multiplier: float = 6.0,
):
    """Save and plot overlaid folding curves for multiple conditions.

    The x-axis is capped at xlim_multiplier times the maximum folding
    time observed in the `xlim_reference_condition` condition (default:
    trap_blocked). This intentionally does NOT extend the x-axis out to
    where the other condition(s) (e.g. trap_involved) reach 100% folded,
    since that timescale is typically much longer and would compress the
    blocked-condition curve into an unreadable sliver near the origin.
    """
    if len(results_list) < 2:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.1))

    combined_rows = []
    reference_max_time_us = None
    for result in results_list:
        condition = result.get("condition_name", "condition")
        time_us = np.asarray(result.get("folding_curve_time_us", []), dtype=float)
        folded_percent = np.asarray(result.get("folding_curve_percent", []), dtype=float)
        if len(time_us) == 0:
            continue
        ax.plot(time_us, folded_percent, linewidth=2, label=condition)
        for t, p in zip(time_us, folded_percent):
            combined_rows.append((condition, t, p))
        if condition == xlim_reference_condition:
            reference_max_time_us = float(np.max(time_us))

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("% folded")
    ax.set_ylim(0, 100)

    if reference_max_time_us is not None:
        x_upper = reference_max_time_us * xlim_multiplier
        ax.set_xlim(0, x_upper)
        print(
            f"Comparison plot x-axis capped at {xlim_multiplier}x the max "
            f"'{xlim_reference_condition}' folding time: 0 to {x_upper:.4f} us "
            f"(max '{xlim_reference_condition}' time: {reference_max_time_us:.4f} us)."
        )
    else:
        print(
            f"WARNING: condition '{xlim_reference_condition}' not found among "
            "results or has no successful folding events; falling back to auto x-axis."
        )
        ax.set_xlim(left=0)

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_png = f"{output_prefix}_comparison_folding_curve.png"
    out_pdf = f"{output_prefix}_comparison_folding_curve.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {out_png}")
    print(f"Saved comparison plot: {out_pdf}")

    out_txt = f"{output_prefix}_comparison_folding_curve.txt"
    with open(out_txt, "w") as f:
        f.write("Condition\tTime_us\tFolded_percentage\n")
        for condition, t, p in combined_rows:
            f.write(f"{condition}\t{t:.8f}\t{p:.4f}\n")
    print(f"Saved comparison data: {out_txt}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-consistent Gillespie folding kinetics on a 2D tIC PMF grid."
    )

    parser.add_argument("--pmf", default=None, help="Single WHAM PMF_2d.out file for one-condition mode.")
    parser.add_argument("--pmf_involved", default=None, help="Unmasked PMF_2d.out for trap-involved condition.")
    parser.add_argument("--pmf_blocked", default=None, help="Pre-masked PMF_2d_masked.out for trap-blocked condition.")
    parser.add_argument(
        "--diffusion_pkl",
        default=None,
        help=(
            "Pickle file containing position-dependent D(tIC1,tIC2). "
            "Expected keys: diffusion_2d, diffusion, D, or diffusion_tica1 + diffusion_tica2."
        ),
    )
    parser.add_argument(
        "--uniform_D",
        type=float,
        default=1e-11,
        help="Uniform D in tIC^2/ps used only if --diffusion_pkl is omitted.",
    )
    parser.add_argument("--invalid_value", type=float, default=9999999.0)

    parser.add_argument("--temperature_K", type=float, default=310.0)
    parser.add_argument("--pmf_units", choices=["kcal", "kJ"], default="kcal")
    parser.add_argument("--connectivity", choices=["4-neighbor", "8-neighbor"], default="8-neighbor")

    parser.add_argument("--n_trajectories", type=int, default=1000)
    parser.add_argument("--max_time_ps", type=float, default=1e30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track_trajectories", action="store_true")

    # Default folded/unfolded definitions copied from the old workflow.
    parser.add_argument("--folded_tic1", type=float, nargs=2, default=[-1.2, -1.0])
    parser.add_argument("--folded_tic2", type=float, nargs=2, default=[-0.3, 0.3])
    parser.add_argument("--unfolded_tic1", type=float, nargs=2, default=[2.7, 3.0])
    parser.add_argument("--unfolded_tic2", type=float, nargs=2, default=[0.5, 0.7])
    parser.add_argument("--folded_pmf_max", type=float, default=10.0)
    parser.add_argument("--unfolded_pmf_min", type=float, default=5.0)

    parser.add_argument("--block_state_B", action="store_true")
    parser.add_argument("--B", type=float, nargs=2, default=[1.30, 1.35])
    parser.add_argument("--state_radius", type=float, default=0.35)

    parser.add_argument("--condition_name", default=None)
    parser.add_argument("--involved_condition_name", default="trap_involved")
    parser.add_argument("--blocked_condition_name", default="trap_blocked")
    parser.add_argument("--output_prefix", default="gillespie_kinetics")

    return parser.parse_args()


def main():
    args = parse_args()

    common_kwargs = dict(
        diffusion_pkl=args.diffusion_pkl,
        invalid_value=args.invalid_value,
        uniform_D=args.uniform_D,
        folded_tic1_range=tuple(args.folded_tic1),
        folded_tic2_range=tuple(args.folded_tic2),
        unfolded_tic1_range=tuple(args.unfolded_tic1),
        unfolded_tic2_range=tuple(args.unfolded_tic2),
        folded_pmf_max=args.folded_pmf_max,
        unfolded_pmf_min=args.unfolded_pmf_min,
        temperature_K=args.temperature_K,
        pmf_units=args.pmf_units,
        connectivity=args.connectivity,
        n_trajectories=args.n_trajectories,
        max_time_ps=args.max_time_ps,
        seed=args.seed,
        track_trajectories=args.track_trajectories,
        output_prefix=args.output_prefix,
        state_B_center=tuple(args.B),
        state_radius=args.state_radius,
    )

    # Two-PMF mode: reproduce the original workflow where the trap-blocked
    # condition is represented by a separate pre-masked PMF file.  In this
    # mode we do not additionally apply --block_state_B in code.
    if args.pmf_involved is not None or args.pmf_blocked is not None:
        if args.pmf_involved is None or args.pmf_blocked is None:
            raise ValueError("Two-PMF mode requires both --pmf_involved and --pmf_blocked.")

        if args.block_state_B:
            print(
                "WARNING: --block_state_B was provided, but two-PMF mode assumes "
                "the blocked PMF is already pre-masked. Ignoring --block_state_B."
            )

        results_involved = run_condition(
            pmf_file=args.pmf_involved,
            condition_name=args.involved_condition_name,
            block_state_B=False,
            **common_kwargs,
        )

        # Use a different seed for the second condition to avoid identical
        # random streams while preserving reproducibility.
        blocked_kwargs = dict(common_kwargs)
        blocked_kwargs["seed"] = args.seed + 1
        results_blocked = run_condition(
            pmf_file=args.pmf_blocked,
            condition_name=args.blocked_condition_name,
            block_state_B=False,
            **blocked_kwargs,
        )

        save_comparison_curve(
            [results_involved, results_blocked],
            output_prefix=args.output_prefix,
            xlim_reference_condition=args.blocked_condition_name,
            xlim_multiplier=6.0,
        )
        print("=== Done: two-PMF trap-involved/trap-blocked analysis ===")
        return

    # Single-PMF mode.
    if args.pmf is None:
        raise ValueError("Provide either --pmf for single-PMF mode, or both --pmf_involved and --pmf_blocked.")

    condition_name = args.condition_name
    if condition_name is None:
        condition_name = "trap_blocked" if args.block_state_B else "trap_involved"

    run_condition(
        pmf_file=args.pmf,
        condition_name=condition_name,
        block_state_B=args.block_state_B,
        **common_kwargs,
    )

    print("=== Done: single-PMF analysis ===")


if __name__ == "__main__":
    main()
