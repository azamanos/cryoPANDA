import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from utils.utils import coordinates_to_dmatrix

def remove_and_refine_duplicates(predicted_coords, predicted_weights, dist_threshold, batch_size=50):
    """
    Remove duplicate candidate picks and refine surviving coordinates via
    weighted average within each neighbourhood.

    Parameters
    ----------
    predicted_coords : np.ndarray, shape (N, 2)
        Candidate pick coordinates.
    predicted_weights : np.ndarray, shape (N,)
        Confidence scores for each candidate.
    dist_threshold : float
        Neighbourhood radius; candidates closer than this are considered duplicates.
    batch_size : int
        Number of coordinates processed per batch to limit memory usage.

    Returns
    -------
    refined_coords  : np.ndarray, shape (N, 2)
    refined_weights : np.ndarray, shape (N,)
    to_delete       : np.ndarray  — indices of duplicates to remove.
    """
    refined_coords  = np.copy(predicted_coords)
    refined_weights = np.copy(predicted_weights)
    n               = len(predicted_coords)
    batch_loop      = np.linspace(0, n, int(np.ceil(n / batch_size)) + 1, dtype=int)
    to_delete       = []

    for b_i, batch_start in enumerate(batch_loop[:-1]):
        sl      = slice(batch_start, batch_loop[b_i + 1])
        d_mat   = coordinates_to_dmatrix(predicted_coords[sl], predicted_coords)
        dups    = np.unique(np.where(d_mat < dist_threshold)[0])

        for i, j in zip(dups, dups + batch_start):
            closeby   = np.where(d_mat[i] < dist_threshold)[0]
            closeby_r = np.delete(closeby, np.argwhere(closeby == j)[0])

            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                to_delete += closeby_r.tolist()
                region_w   = predicted_weights[closeby]
                refined_coords[j] = np.sum(
                    predicted_coords[closeby] * np.expand_dims(region_w / region_w.sum(), 1), axis=0
                )
            else:
                to_delete.append(j)

    return refined_coords, refined_weights, np.unique(to_delete)


def remove_duplicates(predicted_coords, predicted_weights, dist_threshold, batch_size=50):
    """
    Remove duplicate candidate picks, keeping the highest-scoring one in each
    neighbourhood (no coordinate refinement).

    Parameters
    ----------
    predicted_coords : np.ndarray, shape (N, 2)
    predicted_weights : np.ndarray, shape (N,)
    dist_threshold : float
    batch_size : int

    Returns
    -------
    to_delete : np.ndarray — indices of duplicates to remove.
    """
    n          = len(predicted_coords)
    batch_loop = np.linspace(0, n, int(np.ceil(n / batch_size)) + 1, dtype=int)
    to_delete  = []

    for b_i, batch_start in enumerate(batch_loop[:-1]):
        sl    = slice(batch_start, batch_loop[b_i + 1])
        d_mat = coordinates_to_dmatrix(predicted_coords[sl], predicted_coords)
        dups  = np.unique(np.where(d_mat < dist_threshold)[0])

        for i, j in zip(dups, dups + batch_start):
            closeby   = np.where(d_mat[i] < dist_threshold)[0]
            closeby_r = np.delete(closeby, np.argwhere(closeby == j)[0])
            if (predicted_weights[j] > predicted_weights[closeby_r]).all():
                to_delete += closeby_r.tolist()
            else:
                to_delete.append(j)

    return np.unique(to_delete)


def pick_particles_from_pca_projection(pca_projected, rss_x, rss_y, n_components,
                                        H, W, stride, picking_radius, m_shape, im_size,
                                        cap=0.):
    """
    Convert a PCA projection grid into particle coordinates.

    Parameters
    ----------
    pca_projected : np.ndarray, shape (N, n_components)
    rss_x, rss_y  : int  — grid dimensions
    H, W          : int  — original micrograph dimensions
    stride        : int
    picking_radius: float — duplicate-removal radius in grid units
    m_shape       : tuple — (H, W) of the full micrograph
    im_size       : int   — patch side length
    cap           : float — minimum normalised PCA score to keep as candidate

    Returns
    -------
    p_coords : np.ndarray, shape (M, 2)  — picked coordinates in full-res pixels
    w_coords : np.ndarray, shape (M,)    — associated scores
    """
    grid = (
        torch.from_numpy(pca_projected)
        .reshape(rss_x, rss_y, n_components)
        .permute(2, 0, 1).unsqueeze(0).float()
    )
    grid = (grid - grid.min()) / (grid.max() - grid.min())

    tp = ((H // stride) - rss_x) // 2
    lr = ((W // stride) - rss_y) // 2
    if tp or lr:
        grid = F.pad(grid, (lr, lr, tp, tp), mode='constant', value=0)
    grid = grid[0, 0]   # (grid_H, grid_W)

    # Threshold and collect candidates
    indxs   = np.where(grid > cap)
    weights = grid[indxs].detach().numpy()

    # Refine and remove close duplicates in grid space
    refined_coords, refined_weights, delete = remove_and_refine_duplicates(
        np.array(indxs).T.astype(float), weights, picking_radius
    )

    # Scale to full micrograph pixel space
    p_coords = np.round(
        np.delete(refined_coords, delete, axis=0)
        * np.array(m_shape) / np.array(grid.shape)
    )
    w_coords = np.delete(refined_weights, delete, axis=0)

    # Second pass: remove remaining duplicates in pixel space
    delete2 = remove_duplicates(p_coords, w_coords, im_size // 6)
    if len(delete2):
        p_coords = np.round(np.delete(p_coords, delete2, axis=0)).astype(int)
        w_coords = np.delete(w_coords, delete2, axis=0)

    return p_coords, w_coords


def visualize_picks(p_coords, m_shape, im_size, out_path):
    """
    Render picked coordinates as dilated blobs on a blank canvas and save as PNG.

    Parameters
    ----------
    p_coords : np.ndarray, shape (M, 2)
    m_shape  : tuple  — (H, W)
    im_size  : int    — patch side length; blob radius = im_size // 10
    out_path : str
    """
    canvas = np.zeros(m_shape)
    canvas[tuple(p_coords.T)] = 1
    canvas = torch.from_numpy(canvas)[None, None].float()

    kernel = torch.ones((1, 1, im_size // 10, im_size // 10))
    canvas = torch.nn.functional.conv2d(
        canvas, kernel, padding=kernel.shape[-1] // 2
    )[0, 0].detach().numpy()

    Image.fromarray((canvas * 255).astype(np.uint8), mode='L').save(out_path)
