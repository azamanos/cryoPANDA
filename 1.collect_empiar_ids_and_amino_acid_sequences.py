#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
from utils.scraping_utils import fetch_empiar_emd_metadata_and_pdb_data_multiprocess


# In[2]:


# ---------------------------------------------------------------------------
# Read EMPIAR CSV metadata for particles and single-frame micrographs
# ---------------------------------------------------------------------------

# Open metadata from EMPIAR for particles
with open('./metadata/EMPIAR_search_results_particles.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_info_particles = []
    for row in reader:
        csv_info_particles.append(row)

csv_info_particles = np.array(csv_info_particles)

# Open metadata from EMPIAR for single-frame micrographs
with open('./metadata/EMPIAR_search_results_single_frame.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_info_sfm = []
    for row in reader:
        # Skip entries already present in the particles CSV
        if row[0] in csv_info_particles[1:][:, 0]:
            continue
        csv_info_sfm.append(row)

csv_info_sfm = np.array(csv_info_sfm)

print(f"Total number of entries {len(csv_info_particles[1:]) + len(csv_info_sfm[1:])}")

csv_info = np.concatenate((csv_info_particles, csv_info_sfm[1:]))


# In[3]:


# ---------------------------------------------------------------------------
# Build EMPIAR -> [EMD] mapping
# ---------------------------------------------------------------------------

empiar_emd_dict = {}

for entry in csv_info[1:]:
    empiar = entry[0].split('-')[-1]
    emd = entry[10].split(',')

    # len(emd) - 1 is truthy iff there is more than one element
    if len(emd) - 1:
        emd = [i.split('-')[-1] for i in emd]
    else:
        emd = [emd[0].split('-')[-1]]

    empiar_emd_dict[empiar] = emd

all_categories = (
    'micrographs - single frame',
    'picked particles - single frame - unprocessed',
    'picked particles - single frame - processed',
    'picked particles - multiframe - processed',
    'picked particles - multiframe - unprocessed',
)

info = fetch_empiar_emd_metadata_and_pdb_data_multiprocess(
    empiar_emd_dict,
    10,
    use_local=True,
    category_filter=all_categories,
    download_pdb=False
)

# As above, this is a no-op in a script but handy when run interactively
len(info)


# In[4]:


# ---------------------------------------------------------------------------
# Build info_dict from particle and single-frame metadata
# ---------------------------------------------------------------------------

info_dict = {}

for item in info:
    if not item:
        continue
    try:
        empiar_id = list(item.keys())[0]
        info_dict[empiar_id] = list(item.values())[0]
    except Exception:
        # Skip malformed entries
        continue

len(info_dict.keys())


# ---------------------------------------------------------------------------
# FASTA maker
# ---------------------------------------------------------------------------

fasta = []

for empiar_id, empiar_data in info_dict.items():
    for emd_id, emd_data in empiar_data["emd_info"].items():
        for mol in emd_data["macromolecules"]:
            if not len(mol["sequence"]):
                continue

            fasta.append(
                f">{empiar_id}_{emd_id}_{mol['macromolecule_id']}\n"
            )
            fasta.append(f"{mol['sequence']}\n")

with open("./metadata/empiar_seq.fasta", "w") as f:
    for line in fasta:
        f.write(line)

len(fasta)

