import torch
import mrcfile
import numpy as np
from PIL import Image
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dinov3.eval.linear import create_linear_input
from utils.utils import MRC, normalize_single_image_array

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
