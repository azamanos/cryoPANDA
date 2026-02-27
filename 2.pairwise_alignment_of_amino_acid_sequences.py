#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Bio import SeqIO, pairwise2
from itertools import combinations
from multiprocessing import Pool, cpu_count


# In[ ]:


# ---------------------------------------------------------------------------
# Function to compute % identity between two records
# ---------------------------------------------------------------------------

def percent_identity(args):
    """
    Compute percent sequence identity between two Biopython SeqRecord objects.

    Parameters
    ----------
    args : tuple
        (rec1, rec2) where each is a SeqRecord.

    Returns
    -------
    tuple or None
        (rec1.id, rec2.id, pid) where pid is percent identity (float),
        or None if the pair is skipped based on ID prefix.
    """
    rec1, rec2 = args

    # Optional filter: skip sequences with same prefix (e.g., EMPIAR dataset handling)
    if rec1.id[:5] == rec2.id[:5]:
        return None

    alignments = pairwise2.align.globalxx(rec1.seq, rec2.seq)
    top = alignments[0]

    matches = sum(a == b for a, b in zip(top.seqA, top.seqB))
    pid = 100.0 * matches / len(top.seqA)

    return rec1.id, rec2.id, pid


# ---------------------------------------------------------------------------
# Main computation: load sequences, compute pairwise identity, save results
# ---------------------------------------------------------------------------

# Load sequences from FASTA
fasta_file = "./metadata/empiar_seq.fasta"
sequences = list(SeqIO.parse(fasta_file, "fasta"))

# Create all unique combinations (unordered pairs)
sequence_pairs = list(combinations(sequences, 2))

# Use multiprocessing Pool
threads = cpu_count()  # Or set manually (e.g., threads = 4)
with Pool(processes=threads) as pool:
    results = pool.map(percent_identity, sequence_pairs)

# Filter out skipped (None) entries
pairwise_alignment = [res for res in results if res is not None]

# Save to .npy file
pairwise_array = np.array(pairwise_alignment, dtype=object)
np.save("./metadata/pairwise_alignment_empiar.npy", pairwise_array)


# In[2]:


# ------------------------------------------------------------------
# Filter pairwise % identity results and extract unique EMPIAR IDs.
# ------------------------------------------------------------------
percentage_similarity = 30

# Load pairwise alignment array
pairwise_alignment = np.load("./metadata/pairwise_alignment_empiar.npy", allow_pickle=True)

# ---------------------------------------------------------------------------
# Extract all EMPIAR IDs appearing in any alignment pair
# ---------------------------------------------------------------------------

empiar_ids = []

for entry in pairwise_alignment:
    rec1_id, rec2_id, pid = entry
    empiar_ids.append(rec1_id.split("_")[0])
    empiar_ids.append(rec2_id.split("_")[0])

empiar_ids = np.unique(empiar_ids)

# ---------------------------------------------------------------------------
# Filter pairs with percent identity above threshold
# ---------------------------------------------------------------------------

pairwise_alignment = pairwise_alignment[
    np.where(pairwise_alignment[:, -1] > percentage_similarity)[0]
]


# In[3]:


# ----------------------------------------------------------------------------------------------------
# Build unique EMPIAR/EMD pair matches from pairwise_alignment and store them in a lookup dictionary.
# ----------------------------------------------------------------------------------------------------

unique_matches = []

for ip, pairs in enumerate(pairwise_alignment):
    pair_1, pair_2, perc = pairs

    # IDs are formatted like "EMPIAR_EMD_macromolecule"
    empiar_p1, emd_p1 = pair_1.split("_")[:-1]
    empiar_p2, emd_p2 = pair_2.split("_")[:-1]

    umta = np.array(unique_matches)

    if ip:
        # Find rows where [pair_1, pair_2] matches
        fm = np.where(
            (umta[:, 0] == empiar_p1) & (umta[:, 2] == empiar_p2)
        )[0]

        # Find rows where [pair_2, pair_1] matches (symmetric pair)
        sm = np.where(
            (umta[:, 0] == empiar_p2) & (umta[:, 2] == empiar_p1)
        )[0]

        if len(fm):
            fm = int(fm[0])
            if perc > unique_matches[fm][-1]:
                unique_matches[fm][-1] = perc
            continue

        if len(sm):
            sm = int(sm[0])
            if perc > unique_matches[sm][-1]:
                unique_matches[sm][-1] = perc
            continue

    # New unique pair: store EMPIAR1, EMD1, EMPIAR2, EMD2, perc
    unique_matches.append([empiar_p1, emd_p1, empiar_p2, emd_p2, perc])

# Convert list to numpy array
unique_matches = np.array(unique_matches)

# ---------------------------------------------------------------------------
# Build dictionary: each key is "EMPIAR_EMD" pointing to all partners
# ---------------------------------------------------------------------------

unique_matches_dict = {}

for ent in unique_matches:
    empiar1, emd1, empiar2, emd2, perc = ent

    key_1 = f"{str(empiar1)}_{str(emd1)}"
    key_2 = f"{str(empiar2)}_{str(emd2)}"

    # Add pair (2 -> 1) under key_1
    if key_1 in unique_matches_dict:
        unique_matches_dict[key_1].append([empiar2, emd2, perc])
    else:
        unique_matches_dict[key_1] = [[empiar2, emd2, perc]]

    # Add pair (1 -> 2) under key_2
    if key_2 in unique_matches_dict:
        unique_matches_dict[key_2].append([empiar1, emd1, perc])
    else:
        unique_matches_dict[key_2] = [[empiar1, emd1, perc]]

# Convert lists to numpy arrays
unique_matches_dict = {
    k: np.array(v) for k, v in unique_matches_dict.items()
}

len(unique_matches_dict)


# In[4]:


"""
Extract entries from unique_matches_dict that have 4 or more partners.
"""

unique_matches_dict_multiple = {}

for k, v in unique_matches_dict.items():
    if len(v) >= 4:
        unique_matches_dict_multiple[k] = v

len(unique_matches_dict_multiple.keys())


# In[5]:


# ---------------------------------------------------------------------------
# Static selected EMPIAR ID lists
# ---------------------------------------------------------------------------

micrograph_entries = (
    '10004', '10005', '10012', '10017', '10025', '10056', '10061', '10075', '10122', '10160',
    '10175', '10189', '10190', '10192', '10193', '10199', '10203', '10208', '10217', '10240',
    '10268', '10271', '10283', '10289', '10290', '10291', '10379', '10401', '10411', '10425',
    '10433', '10467', '10475', '10489', '10519', '10560', '10590', '10652', '10667', '10706',
    '10707', '10735', '10760', '10790', '10794', '10800', '10882', '10888', '10890', '10891',
    '10893', '10919', '10920', '10930', '11060', '11131', '11139', '11146', '11149', '11167',
    '11193', '11194', '11204', '11267', '11268', '11341', '11374', '11443', '11501', '11524',
    '11553', '11556', '11567', '11604', '11605', '11607', '11608', '11615', '11703', '11719',
    '11726', '11755', '11761', '11763', '11768', '11789', '11795', '11840', '11847', '11954',
    '11986', '12087', '12106', '12140', '12143', '12171', '12237', '12240', '12310', '12311',
    '12486', '12561', '12562', '12667',
)

particle_entries = (
    '10024', '10028', '10044', '10049', '10059', '10063', '10072', '10076', '10081', '10090',
    '10091', '10093', '10096', '10097', '10099', '10107', '10123', '10124', '10127', '10128',
    '10166', '10176', '10180', '10202', '10229', '10254', '10264', '10278', '10279', '10285',
    '10294', '10299', '10307', '10308', '10309', '10317', '10328', '10330', '10333', '10335',
    '10336', '10341', '10342', '10344', '10345', '10347', '10350', '10357', '10358', '10373',
    '10374', '10380', '10391', '10395', '10396', '10397', '10398', '10399', '10406', '10407',
    '10409', '10420', '10421', '10437', '10443', '10454', '10455', '10465', '10470', '10481',
    '10482', '10483', '10532', '10536', '10640', '10659', '10669', '10697', '10703', '10722',
    '10739', '10752', '10770', '10786', '10792', '10810', '10841', '10873', '10874', '10876',
    '11000', '11005', '11043', '11120', '11128', '11211', '11233', '11247', '11270', '11283',
    '11362', '11521', '11522', '11523', '11526', '11618', '11665', '11681', '11706', '11720',
    '11734', '11762', '11791', '11792', '11796', '11797', '11834', '11836', '11844', '11898',
    '11910', '11925', '11987', '12036', '12093', '12094', '12097', '12121', '12180', '12443',
    '12444', '12510',
)

total_entries = micrograph_entries + particle_entries

# This line is a no-op in a script but useful when run interactively.
len(particle_entries), len(micrograph_entries), len(total_entries)

