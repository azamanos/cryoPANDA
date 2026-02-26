# cryoPANDA

A toolkit for building cryo-EM protein structure datasets for machine learning research. cryoPANDA automates the collection of metadata and image data from [EMPIAR](https://www.ebi.ac.uk/empiar/) and [EMDB](https://www.ebi.ac.uk/emdb/), aligns density maps, and prepares train/eval splits for linear probing with DINOv2 features.

## Requirements

- Python 3.8+
- [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) (for volume alignment — must be installed separately)
- See `requirements.txt` for Python package dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/azamanos/cryoPANDA.git
cd cryoPANDA

# Create and activate a conda environment
conda create -n cryopanda python=3.10
conda activate cryopanda

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Set the ChimeraX path if not at /usr/bin/chimerax
export CHIMERAX_PATH=/path/to/chimerax
```

## Pipeline Overview

The data processing pipeline is implemented as a series of numbered Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `1.collect_empiar_ids_and_amino_acid_sequences.ipynb` | Scrapes EMPIAR/EMDB metadata and amino acid sequences |
| `2.pairwise_alignment_of_amino_acid_sequences.ipynb` | Pairwise sequence alignment to detect redundancy |
| `3.collect_statistics.ipynb` | Computes structural and experimental statistics |
| `4.collect_particle_annotations_from_h5_files.ipynb` | Extracts particle annotations from HDF5 files |
| `5.split_dataset_in_train_eval_sets.ipynb` | Splits data into train/eval sets |
| `6.create_linear_probing_sets.ipynb` | Creates linear probing datasets using DINOv2 features |

Run the notebooks in order. Each notebook reads outputs produced by the previous one.

## Utility Modules

| Module | Description |
|--------|-------------|
| `utils/scraping_utils.py` | Fetch EMPIAR and EMDB XML metadata; download PDB assemblies |
| `utils/download_utils.py` | Parallel download of EMDB map files |
| `utils/utils.py` | PDB/CIF and MRC file parsers; geometric computation utilities |
| `utils/volumes.py` | ChimeraX-based volume alignment wrapper |
| `utils/align.py` | ChimeraX alignment script (invoked as a subprocess) |
| `utils/resize.py` | Image resizing utility |

## Configuration

The DINOv2 linear probing configuration files are located in `metadata/DINOv2_checkpoint/`. Before running the pipeline, update the dataset and output paths in `config.yaml` to match your local file system:

```yaml
dataset_path: /path/to/your/h5/data
output_dir: /path/to/your/output/directory
```

## License

MIT License — see [LICENSE](LICENSE) for details.
