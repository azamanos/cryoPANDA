import os
import json
import h5py
import numpy as np
import pandas as pd

import umap.umap_ as umap
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from dinov3.eval.linear import create_linear_input

def load_h5_particles(h5_path):
    """
    Load particle images and annotations from an H5 file.

    Parameters
    ----------
    h5_path : str

    Returns
    -------
    particles      : np.ndarray, shape (N, H, W)  — uint8 particle images
    annotations_df : pd.DataFrame
    """
    with h5py.File(h5_path, 'r') as f:
        particles        = f['images'][:]
        annotations_json = f['annotations'][()].decode()
    annotations_df = pd.DataFrame.from_dict(json.loads(annotations_json))
    return particles, annotations_df


def preprocess_particles(particles, mean, std):
    """
    Normalise particle images to [0,1], apply dataset standardisation,
    and return a float32 tensor of shape (N, 1, H, W).
    """
    p = particles / 255.0
    p = (p - mean) / std
    return torch.from_numpy(p).unsqueeze(1).float()


def extract_features_per_patch(patches, backbone, n_layers, device, batch_size, autocast_dtype):
    """
    Extract per-patch token features from the last ViT layer.

    Returns
    -------
    np.ndarray, shape (N, patch_h, patch_w, D)
    """
    N, latents = patches.shape[0], []
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
            for start in range(0, N, batch_size):
                batch = patches[start : start + batch_size].to(device)
                feats = (
                    backbone.get_intermediate_layers(
                        batch, n=range(n_layers), reshape=True, norm=True
                    )[-1]
                    .cpu().float().permute(0, 2, 3, 1).numpy()
                )
                latents.append(feats)
                print(f'  {start + len(feats)}/{N}', end='\r')
    print()
    return np.concatenate(latents)  # (N, patch_h, patch_w, D)


def extract_features(patches, feature_model, n_last_blocks, device, batch_size, autocast_dtype, label=''):
    """
    Extract CLS-level features via the linear-probing feature model.

    Returns
    -------
    np.ndarray, shape (N, D)
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


def run_pca(latents, n_components):
    """
    Fit PCA on *latents* and return the projected coordinates.

    Returns
    -------
    np.ndarray, shape (N, n_components)
    """
    pca = PCA(n_components=n_components, whiten=True).fit(latents)
    return pca.transform(latents)


def get_empiar_ids(annotations):
    """
    Extract clean EMPIAR IDs from an annotations DataFrame.

    Returns
    -------
    list[str]
    """
    raw = np.unique(annotations['EMPIAR_ID'].astype(str))
    return [i.split('.')[0].split('_')[0] for i in raw]


def run_patch_pca_per_experiment(latents, empiar_ids, sample_num, n_patches, pca_components):
    """
    Fit a per-experiment PCA on patch-level latents.

    Parameters
    ----------
    latents       : np.ndarray, shape (N_total, patch_h, patch_w, D)
    empiar_ids    : list[str]
    sample_num    : int  — particles per experiment
    n_patches     : int  — total patches per image (e.g. 196 = 14×14)
    pca_components: int

    Returns
    -------
    list[np.ndarray]  — one (sample_num, n_patches, pca_components) array per experiment
    """
    results = []
    for i in range(len(empiar_ids)):
        exp = latents[i*sample_num:(i+1)*sample_num].reshape(sample_num * n_patches, -1)
        projected = run_pca(exp, pca_components).reshape(sample_num, n_patches, -1)
        results.append(projected)
    return results


def save_particle_and_pca_overlay(particles, projected, exp_idx, ridx,
                                   eid, sample_num, n_patches, pca_components,
                                   im_size, out_dir):
    """
    Save the raw particle image and its PCA patch-token overlay for one random particle.

    Parameters
    ----------
    particles     : np.ndarray, shape (N_total, H, W)  — uint8
    projected     : np.ndarray, shape (sample_num, n_patches, pca_components)
    exp_idx       : int  — experiment index within the particle set
    ridx          : int  — index of the particle to visualise within the experiment
    eid           : str  — EMPIAR ID string used for file naming
    """
    # Save raw particle
    Image.fromarray(particles[exp_idx * sample_num + ridx]).convert('L').save(
        os.path.join(out_dir, f'{eid}_{ridx}_particle.png')
    )

    # Upsample 14×14 PCA grid to im_size
    patch_side = int(n_patches ** 0.5)
    proj_grid  = (
        torch.from_numpy(projected[ridx].reshape(patch_side, patch_side, pca_components))
        .permute(2, 0, 1).unsqueeze(0)
    )
    proj_up = F.interpolate(proj_grid, size=(im_size, im_size), mode='nearest')[0]\
               .permute(1, 2, 0).numpy()

    p_min = proj_up.min(axis=(0, 1), keepdims=True)
    p_max = proj_up.max(axis=(0, 1), keepdims=True)
    u8    = ((proj_up - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)

    Image.fromarray(u8, mode='RGB').save(
        os.path.join(out_dir, f'{eid}_{ridx}_particle_PCA_{pca_components}.png')
    )


def fit_umap(X, n_neighbors, min_dist, metric, seed):
    """
    Fit a 2-D UMAP embedding on *X*.

    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def plot_umap(X_umap, labels, empiar_ids, title, out_path, dpi=150):
    """
    Scatter plot of a 2-D UMAP embedding coloured by experiment, with
    EMPIAR ID labels at each cluster centroid.

    Parameters
    ----------
    X_umap     : np.ndarray, shape (N, 2)
    labels     : np.ndarray, shape (N,)  — integer experiment indices
    empiar_ids : list[str]
    title      : str
    out_path   : str
    """
    centroids = np.array([X_umap[labels == k].mean(axis=0) for k in range(len(empiar_ids))])
    cmap      = plt.colormaps['jet'].resampled(len(empiar_ids))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=5, cmap=cmap, alpha=0.7)

    for k, eid in enumerate(empiar_ids):
        cx, cy = centroids[k]
        ax.text(
            cx, cy, str(eid),
            fontsize=11, weight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='black', alpha=0.8),
        )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.show()
    print(f'Saved → {out_path}')