import os
import time
import torch
import mrcfile
import numpy as np
from PIL import Image
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dinov3.eval.linear import create_linear_input
from utils.utils import MRC, normalize_single_image_array
from utils.picking import remove_and_refine_duplicates, remove_duplicates

def load_micrograph(mrc_path):
    """
    Load a micrograph as a float64 numpy array.
    Falls back to the custom MRC reader for non-standard files.
    """
    try:
        with mrcfile.open(mrc_path, 'r') as f:
            return f.data.astype(float)
    except Exception:
        return MRC(mrc_path).data[:, :, 0].astype(float)


def save_normalised_png(m, out_path):
    """Normalise a micrograph array to [0, 255] and save as a grayscale PNG."""
    m_norm = normalize_single_image_array(m)
    Image.fromarray((m_norm * 255).astype(np.uint8), mode='L').save(out_path)


def extract_patches(m, im_size, stride):
    """
    Extract a centered sliding-window patch grid from micrograph *m*.

    Returns
    -------
    patches : torch.Tensor  shape (N, im_size, im_size)
    rss_x   : int  number of grid rows
    rss_y   : int  number of grid columns
    """
    H, W     = m.shape
    rss_x    = (H - im_size) // stride + 1
    rss_y    = (W - im_size) // stride + 1
    grid_h   = (rss_x - 1) * stride + im_size
    grid_w   = (rss_y - 1) * stride + im_size
    offset_x = (H - grid_h) // 2
    offset_y = (W - grid_w) // 2

    tiles = [
        m[offset_x + ix * stride : offset_x + ix * stride + im_size,
          offset_y + iy * stride : offset_y + iy * stride + im_size]
        for ix in range(rss_x)
        for iy in range(rss_y)
    ]
    return torch.from_numpy(np.array(tiles)).float(), rss_x, rss_y

def extract_patches_torch(m, im_size, stride):
    m = torch.from_numpy(m)
    H, W = m.shape
    rss_x = (H - im_size) // stride + 1
    rss_y = (W - im_size) // stride + 1
    grid_h = (rss_x - 1) * stride + im_size
    grid_w = (rss_y - 1) * stride + im_size
    offset_x = (H - grid_h) // 2
    offset_y = (W - grid_w) // 2

    cropped = m[offset_x:offset_x + grid_h, offset_y:offset_y + grid_w]
    patches = (cropped.unfold(0, im_size, stride)
                       .unfold(1, im_size, stride)
                       .reshape(-1, im_size, im_size)
                       .contiguous()
                       .float())
    return patches, rss_x, rss_y

def normalise_patches(patches, mean, std):
    """
    Per-patch min-max normalisation, contrast inversion, then dataset standardisation.

    Parameters
    ----------
    patches : torch.Tensor  shape (N, im_size, im_size)

    Returns
    -------
    torch.Tensor  shape (N, 1, im_size, im_size)
    """
    p       = patches.unsqueeze(1)  # (N, 1, H, W)
    p_min   = p.amin(dim=(-1, -2), keepdim=True)
    p_max   = p.amax(dim=(-1, -2), keepdim=True)
    p       = (p - p_min) / (p_max - p_min + 1e-8)
    p       = 1 - p                  # invert contrast
    return (p - mean) / std


def extract_features(patches, feature_model, n_last_blocks, device, batch_size, autocast_dtype, label=''):
    """
    Run batched inference through the DINOv2 backbone.

    Returns
    -------
    np.ndarray  shape (N, D)
    """
    N, latents = patches.shape[0], []
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
            for start in range(0, N, batch_size):
                batch = patches[start : start + batch_size].to(device)
                feats = create_linear_input(
                    feature_model(batch.float()), n_last_blocks, True
                ).cpu().float().numpy()
                latents.append(feats)
                print(f'  {label}  {start + len(feats)}/{N}', end='\r')
    print()
    return np.concatenate(latents)  # (N, D)


def to_fullres(grid_tensor, H, W, rss_x, rss_y, stride, m_shape):
    """
    Pad and upsample a (1, C, rss_x, rss_y) grid to the original micrograph size.
    """
    tp = ((H // stride) - rss_x) // 2
    lr = ((W // stride) - rss_y) // 2
    if tp or lr:
        grid_tensor = F.pad(grid_tensor, (lr, lr, tp, tp), mode='constant', value=0)
    return F.interpolate(grid_tensor, size=m_shape, mode='nearest')[0]


def run_pca(latents, n_components):
    """
    Fit PCA on *latents*, upsample to full resolution.
    """
    pca       = PCA(n_components=n_components, whiten=True).fit(latents)
    projected = pca.transform(latents)  # (N, C)

    return projected

def run_pca_and_save(latents, rss_x, rss_y, H, W, stride, m_shape,
                     n_components, out_dir, emp_id, im_size):
    """
    Fit PCA on *latents*, upsample to full resolution, and save:
      - a single-channel PNG of the 1st component
      - an RGB PNG of the first 3 components
    """

    projected = torch.from_numpy(run_pca(latents, n_components))

    grid    = projected.reshape(rss_x, rss_y, n_components).permute(2, 0, 1).unsqueeze(0).float()
    proj_up = to_fullres(grid, H, W, rss_x, rss_y, stride, m_shape).permute(1, 2, 0).numpy()

    p_min = proj_up.min(axis=(0, 1), keepdims=True)
    p_max = proj_up.max(axis=(0, 1), keepdims=True)
    u8    = ((proj_up - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)

    Image.fromarray(u8[:, :, 0], mode='L').save(
        os.path.join(out_dir, f'{emp_id}_PCA_1_mask_{im_size}_{stride}.png')
    )
    Image.fromarray(u8, mode='RGB').save(
        os.path.join(out_dir, f'{emp_id}_PCA_{n_components}_mask_{im_size}_{stride}.png')
    )


def run_kmeans_and_save(latents, rss_x, rss_y, H, W, stride, m_shape,
                        n_clusters, out_dir, emp_id, im_size):
    """
    Fit K-Means on *latents*, upsample the label map to full resolution, and save
    a grayscale PNG.
    """
    labels = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(latents)

    grid  = (
        torch.from_numpy(labels.reshape(rss_x, rss_y).astype(np.float32))
        .unsqueeze(0).unsqueeze(0)
    )
    km_up = to_fullres(grid, H, W, rss_x, rss_y, stride, m_shape)[0].numpy()

    km_min, km_max = km_up.min(), km_up.max()
    u8 = ((km_up - km_min) / (km_max - km_min + 1e-8) * 255).astype(np.uint8)

    Image.fromarray(u8, mode='L').save(
        os.path.join(out_dir, f'{emp_id}_kmeans_{n_clusters}_centers_{im_size}_{stride}.png')
    )

# ── Functions ─────────────────────────────────────────────────────────────────────
def write_star_file(star_file_name, data_particles):
    """
    Write particle coordinates to a RELION-style STAR file (overwrites existing files).

    Parameters
    ----------
    star_file_name : str
    data_particles : list[str]
        One row per particle: '<MicrographName> <CoordinateX> <CoordinateY>',
        with X the column and Y the row of the micrograph (RELION convention).
    """
    header = [
        '',
        'data_particles',
        '',
        'loop_',
        '_rlnMicrographName #1 ',
        '_rlnCoordinateX #2 ',
        '_rlnCoordinateY #3 ',
    ]
    with open(star_file_name, 'w') as f:
        f.write('\n'.join(header + list(data_particles)) + '\n')


def normalize_array(arr, min_value=0., max_value=1.):
    """
    Min-max normalise *arr* to [min_value, max_value]. A constant array maps to min_value.

    Parameters
    ----------
    arr                  : np.ndarray
    min_value, max_value : float

    Returns
    -------
    np.ndarray — float64 for float64 input, float32 otherwise
    """
    arr = np.asarray(arr)
    if arr.dtype != np.float64:
        arr = arr.astype(np.float32, copy=False)   # integer micrographs would overflow in arr - min
    a_min, a_max = arr.min(), arr.max()
    normalized   = np.zeros_like(arr) if a_max == a_min else (arr - a_min) / (a_max - a_min)
    if (min_value, max_value) != (0., 1.):
        normalized = normalized * (max_value - min_value) + min_value
    return normalized


def create_2d_circular_mask(box_size, diameter, inverse=False):
    """
    Boolean mask of a disk of *diameter* centred in a box of *box_size*.
    For an even box and an even diameter, the two central pixels of each axis are both
    treated as the centre, so the mask is symmetric.

    Parameters
    ----------
    box_size : tuple  — (H, W)
    diameter : float
    inverse  : bool   — if True, return the area outside the disk

    Returns
    -------
    np.ndarray of bool, shape (H, W)
    """
    def centred_axis(n):
        c = n // 2
        if not n % 2 and not diameter % 2:
            return np.concatenate((np.arange(1, c + 1), np.arange(c, n))) - c
        return np.arange(n) - c

    dist2 = centred_axis(box_size[0])[:, None] ** 2 + centred_axis(box_size[1])[None, :] ** 2
    r2    = (diameter / 2) ** 2
    return dist2 >= r2 if inverse else dist2 < r2


def micrograph_mean_and_sigma(micrograph, mask, device='cpu'):
    """
    Local mean and standard deviation of *micrograph* under *mask*, for every position
    of the mask centre, via FFT-based (circular) cross-correlation.

    Parameters
    ----------
    micrograph : np.ndarray or torch.Tensor, shape (H, W)
    mask       : np.ndarray or torch.Tensor, shape (H, W) — centred in the array
    device     : str

    Returns
    -------
    mean  : np.ndarray, shape (H, W)
    sigma : np.ndarray, shape (H, W)
    """
    micrograph = torch.as_tensor(micrograph, device=device)
    mask       = torch.as_tensor(mask, dtype=micrograph.dtype, device=device)
    n_pixels   = mask.sum()

    conj_fft_mask = torch.conj(torch.fft.fft2(mask))
    local_sum     = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(micrograph) * conj_fft_mask)).real
    local_sum_sq  = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(micrograph ** 2) * conj_fft_mask)).real

    mean     = local_sum / n_pixels
    variance = (local_sum_sq / n_pixels - mean ** 2).clamp_min(0)   # rounding can make it slightly < 0
    return mean.cpu().numpy(), variance.sqrt().cpu().numpy()


def relion_normalization(micrograph, normalization_filter, device='cpu'):
    """
    RELION-style background normalisation at every position: subtract the local mean and
    divide by the local standard deviation under *normalization_filter* (e.g. the area
    outside a particle-sized disk), then min-max normalise to [0, 1].

    Parameters
    ----------
    micrograph           : np.ndarray, shape (H, W)
    normalization_filter : np.ndarray, shape (H, W) — background mask centred in the array
    device               : str

    Returns
    -------
    np.ndarray, shape (H, W)
    """
    normalized_micrograph = normalize_array(micrograph)
    mean, sigma = micrograph_mean_and_sigma(normalized_micrograph, normalization_filter, device)
    return normalize_array((normalized_micrograph - mean) / np.maximum(sigma, 1e-6))   # guard flat regions


def sample_candidate_particles(m_list, window, output_path, resize=8, device='cpu'):
    """
    Sample candidate particle positions on each micrograph and write them to a STAR file,
    so that picking can be restricted to these positions.

    Each micrograph is min-max normalised, downsampled by *resize*, and RELION-normalised
    against the corners of a 1.28 × *window* box (outside its inscribed disk). The mean
    density is computed in a circular window of ≈ 1.28 × *window* / 3 on a regular grid,
    and the darkest 1/7 of the grid positions away from the micrograph edges are kept.

    Parameters
    ----------
    m_list      : list[str]  — micrograph .mrc paths
    window      : int        — particle window size in micrograph pixels
    output_path : str        — output STAR file; coordinates in micrograph pixels
    resize      : int        — downsampling factor (1 = none)
    device      : str        — device for the FFT-based normalisation
    """
    # Box sizes in downsampled pixels
    box_final = int(window * 1.28 // resize + window * 1.28 // resize % 2)   # normalisation box (even)
    box_edges = box_final // 1.5                                             # min. distance from the edges
    box       = box_final // 3                                               # scanning window
    bh        = box // 2
    step      = max(box // 4, 1)
    if bh < 1:
        raise ValueError(f'window={window} is too small for resize={resize}; use a smaller resize.')
    wm = create_2d_circular_mask((2 * bh, 2 * bh), 2 * bh)

    star_rows, st = [], time.time()
    for mi, mn in enumerate(m_list):
        with mrcfile.open(mn) as mrc:
            m_ = normalize_array(mrc.data)
        om_shape = np.array(m_.shape)
        if resize != 1:
            new_shape = om_shape // resize + om_shape // resize % 2           # even (H, W)
            m_ = np.array(Image.fromarray(m_).resize((int(new_shape[1]), int(new_shape[0]))))   # PIL takes (W, H)
        H, W = m_.shape

        # RELION-style normalisation against the background outside a disk of box_final
        pad_x, pad_y = (H - box_final) // 2, (W - box_final) // 2
        normalization_filter = np.pad(
            create_2d_circular_mask((box_final, box_final), box_final, inverse=True),
            ((pad_x, H - box_final - pad_x), (pad_y, W - box_final - pad_y)),
        )
        m = relion_normalization(m_, normalization_filter, device)

        # Mean density inside a circular window on a regular grid (row by row)
        xs      = np.linspace(box, H - box, (H - 2 * box) // step).astype(int)
        ys      = np.linspace(box, W - box, (W - 2 * box) // step).astype(int)
        windows = np.lib.stride_tricks.sliding_window_view(m, (2 * bh, 2 * bh))
        means   = np.array([windows[x - bh, ys - bh][:, wm].mean(axis=-1) for x in xs]).ravel()
        coords  = np.stack(np.meshgrid(xs, ys, indexing='ij'), axis=-1).reshape(-1, 2).astype(float)

        # Drop positions closer than box_edges to the edges, then keep the darkest 1/7
        keep          = ((coords >= box_edges) & (coords <= np.array([H, W]) - box_edges)).all(axis=1)
        coords, means = coords[keep], means[keep]
        coords        = coords[np.sort(np.argsort(means, kind='stable')[:len(means) // 7])]

        # Back to micrograph pixels; STAR convention: X = column, Y = row
        coords    *= om_shape / np.array([H, W])
        star_rows += [f'{mn} {r:.2f} {c:.2f}' for r, c in coords]
        print(f'  [{mi + 1}/{len(m_list)}] {(time.time() - st) / 60:.2f} min', end='\r')
    print()

    write_star_file(output_path, star_rows)
    print(f'Saved {len(star_rows):,} candidates → {output_path}')


def pick_particles_from_micrograph(score_map, picking_radius, m_shape, im_size, cap=0.):
    """
    Convert a 2D score map (e.g. a PCA component mapped onto the micrograph) into
    particle coordinates.

    Parameters
    ----------
    score_map      : np.ndarray, shape (h, w)
    picking_radius : float — duplicate-removal radius in score-map pixels
    m_shape        : tuple — (H, W) of the full micrograph; coordinates are scaled from
                     score-map to micrograph pixels (no-op if the map is full resolution)
    im_size        : int   — patch side length; second-pass duplicate radius = im_size // 6
    cap            : float — minimum normalised score to keep as candidate

    Returns
    -------
    p_coords : np.ndarray of int, shape (M, 2) — picked (row, col) in micrograph pixels
    w_coords : np.ndarray, shape (M,)          — associated scores
    """
    grid         = np.asarray(score_map)
    g_min, g_max = grid.min(), grid.max()
    grid         = (grid - g_min) / (g_max - g_min) if g_max > g_min else np.zeros(grid.shape)

    # Threshold and collect candidates
    indxs   = np.where(grid > cap)
    weights = grid[indxs]

    # Refine and remove close duplicates in score-map space
    refined_coords, refined_weights, delete = remove_and_refine_duplicates(
        np.array(indxs).T.astype(float), weights, picking_radius
    )
    delete = delete.astype(int)   # np.unique([]) is float64, which np.delete rejects

    # Scale to full micrograph pixel space
    scale    = np.array(m_shape) / np.array(grid.shape)
    p_coords = np.delete(refined_coords, delete, axis=0) * scale
    w_coords = np.delete(refined_weights, delete, axis=0)

    # Second pass: remove remaining duplicates in pixel space
    delete2  = remove_duplicates(p_coords, w_coords, im_size // 6).astype(int)
    p_coords = np.round(np.delete(p_coords, delete2, axis=0)).astype(int)
    w_coords = np.delete(w_coords, delete2, axis=0)

    return p_coords, w_coords
