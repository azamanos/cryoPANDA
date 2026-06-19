<h1 align="center">cryoPANDA</h1>

cryoPANDA: A benchmark dataset of over 37 million particles from 252 experiments to accelerate automated cryo-EM analysis

<div align="center">
  <img src="./metadata/figures/cryoPANDA.png" width="400">
</div>

## Table of Contents

- [Description](#description)
- [Methods](#methods)
- [Dataset Structure](#dataset_structure)
- [Installation (Linux)](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)
- [Citing this work](#citing-this-work)

<a id="description"></a>
## Description

This repository contains EMPIAR experiments processing and representation-analysis notebooks for the cryoPANDA dataset. The workflow starts from metadata extraction and dataset curation, then moves to linear probing, and unsupervised visual analysis with DINOv2 features.

<a id="methods"></a>
## Methods

<div align="center">
  <img src="./metadata/figures/methods.png" width="600">
</div>

<a id="dataset_structure"></a>
## Dataset Structure

<div align="center">
  <img src="./metadata/figures/cryoPANDA_structure.png" width="900">
</div>

<a id="installation"></a>
## Installation (Linux)

1\. Create the cryoPANDA environment by running the following command in your terminal:
```bash
conda env create -f environment.yml
```
2\. Whenever you want to work on the project, activate the cryoPANDA environment by executing the following command in the terminal:

```bash
conda activate cryoPANDA
```
3\. Download `metadata.zip` from https://doi.org/10.57760/sciencedb.27164 to access all data and pretrained DINOv2 weights required by the notebooks.

<a id="usage"></a>
## Usage

Run notebooks in the following order. The notebooks cover curation of experiments, training linear probing, and analysis.

```bash
jupyter-lab
```

### Notebook pipeline

- **1 · `1_collect_empiar_ids_and_amino_acid_sequences.ipynb`**: Reads EMPIAR metadata CSVs, links EMPIAR→EMD/PDB entries, and exports macromolecule amino-acid sequences.
- **2 · `2_pairwise_alignment_of_amino_acid_sequences.ipynb`**: Computes all-vs-all sequence identity, filters pairs by threshold, and stores sequence-similarity lookup artifacts.
- **3 · `3_collect_statistics.ipynb`**: Aggregates experiment-level metadata/statistics (including manual fixes) and exports a consolidated summary table.
- **4 · `4_collect_particle_annotations_from_h5_files.ipynb`**: Loads per-experiment particle `.h5` files, extracts annotation fields from embedded JSON, and writes a merged annotations table.
- **5 · `5_split_dataset_in_train_eval_sets.ipynb`**: Splits data by experiment (`FileName`) into train/eval partitions to avoid experiment leakage.
- **6 · `6_create_linear_probing_sets.ipynb`**: Builds balanced train/eval subsets per annotation target for linear-probing experiments.
- **7 · `7_train_linear_probing.ipynb`**: Generates and launches linear-probing training commands (frozen DINOv2 backbone + linear head), skipping finished runs.
- **8 · `8_collect_linear_probing_results.ipynb`**: Collects `results-linear.csv` outputs from notebook 7 and compiles summary result tables/plots.
- **9 · `9_pca_and_kmeans_on_micrographs.ipynb`**: Extracts micrograph patches, encodes them with DINOv2, and saves PCA and K-Means visualizations.
- **10 · `10_particle_picking_with_dinov2.ipynb`**: Runs a DINOv2-based feature pipeline for particle picking from micrographs via PCA-projected latent structure.
- **11 · `11_pca_and_umap_on_particles.ipynb`**: Encodes particles with DINOv2, produces per-patch PCA overlays, and computes UMAP visualizations of particle embeddings.

`unzip_data.ipynb`: Finds `.zip` files inside cryoPANDA data subfolders and unpacks them in parallel (requires system `unzip`).

<a id="license"></a>
## License

This project is licensed under the MIT License - see the LICENSE file for details.

<a id="contributing"></a>
## Contributing

If you encounter any issues or have suggestions for improvement, please create an issue on GitHub. We appreciate your contribution!

<a id="contact"></a>
## Contact

For queries and suggestions, please contact: andreas.zamanos@athenarc.gr

<a id="citing-this-work"></a>
## Citing this work

LINK TO PAPER PUBLICATION
