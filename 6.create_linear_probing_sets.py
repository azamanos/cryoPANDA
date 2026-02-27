#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df_ann = pd.read_parquet("./metadata/parquet_files/df_ann.parquet")


# In[3]:


np.random.seed(37)
unique_exp = np.unique(np.array(df_ann['FileName']))
panda_len = len(unique_exp)
perc_eval = 0.15
df_ann_idx = np.arange(panda_len)
np.random.shuffle(df_ann_idx)
eval_empiar_idxs = np.sort(df_ann_idx[:int(panda_len*perc_eval)])
eval_idxs = np.where(np.isin(df_ann['FileIndex'].values, eval_empiar_idxs))[0]
train_empiar_idxs = np.sort(df_ann_idx[int(panda_len*perc_eval):])
train_idxs = np.where(np.isin(df_ann['FileIndex'].values, train_empiar_idxs))[0]


# In[4]:


### Creates particle sets for linear probing training and validation from the DINOv2 pretraining set ###

annotations = ['EMD_Resolution','ReconstructionResolution','MolecularWeight_kDa',\
                'ImagePixelSize','ClassResolution','ClassECA','2DAlignmentClassESS',\
                'DefocusAngle', 'DefocusU', 'DefocusV', 'ProteinClass', 'Symmetry', 'FileName']

bin_dict = {'MolecularWeight_kDa':[20,0,1000],'DefocusU':[10000,-5000,40000],'DefocusV':[10000,-5000,40000],\
            'DefocusAngle':[30,-105, 105], '2DAlignmentClassESS':[1,0.5,5], 'ClassECA':[0.5,0.75,3],\
            'ClassResolution':[3,1.5,26], 'EMD_Resolution':[0.5,1.25,8], 'ReconstructionResolution':[0.5,1.25,8],\
            'ImagePixelSize':[0.25,0.875,3]}   # store results

for ann in annotations:
    #ann = 'DefocusV'
    # --- detect numeric or non-numeric ---
    train_particles = df_ann.loc[train_idxs]
    if ann in ('ProteinClass', 'Symmetry', 'FileName'):
        np_ann = train_particles[ann]
        bin_centers, bin_idx, counts = np.unique(np_ann, return_inverse=True, return_counts=True)
        bin_centers = np.arange(len(bin_centers))
    else:
        if ann == 'ImagePixelSize':
            np_ann = train_particles['ImagePixelSize']*train_particles['ImageSize']/224
        else:
            np_ann = train_particles[ann].astype(float)

        min_v, max_v = np_ann.min(), np_ann.max()        

        if ann in bin_dict:
            bin_dist, start, end = bin_dict[ann]
            bin_edges = np.arange(start, end + bin_dist, bin_dist)
        else:
            bin_dist = 0.5

            # --- compute bin edges only if numeric and not constant ---
            if max_v > min_v:
                start = (np.floor(min_v / bin_dist) * bin_dist)
                end   = ((np.ceil(max_v / bin_dist) * bin_dist))
                start, end = 1.25, 8
                bin_edges = np.arange(start, end + bin_dist, bin_dist)
            else:
                bin_edges = np.array([min_v])  # trivial case
                # Compute bin centers (median of each bin range)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Digitize: assign each sample to a bin index
        bin_idx = np.digitize(np_ann, bin_edges, right=True) - 1
        bin_idx[np.where(bin_idx==-1)] = bin_idx.max()
        #bin_idx = np.clip(bin_idx, 0, len(bin_edges)-2)
        # count per bin (should be roughly equal)
        counts = np.bincount(bin_idx)#, minlength=len(bin_edges)-1)

    num_of_picks_dict = {'ReconstructionResolution':67_000,'EMD_Resolution':60_000,'MolecularWeight_kDa':20_000,\
                     'ImagePixelSize':90_000,'ClassResolution':90_000,'ClassECA':160_000,'2DAlignmentClassESS':160_000,\
                     'DefocusAngle':115_000,'DefocusU':160_000,'DefocusV':160_000,\
                     'ProteinClass':50_000, 'Symmetry':50_000, 'FileName':3_000}#15_000}#3_000}

    np.random.seed(44)
    num_of_picks = num_of_picks_dict[ann]  #(250_000, 350_000, 200_000, 200_000, 350_000, 300_000)
    num_of_picks_eval = int(0.2*(num_of_picks_dict[ann]))  #(250_000, 350_000, 200_000, 200_000, 350_000, 300_000)
    # Collect indices for all bins
    total_particle_idxs, total_particle_idxs_eval = [], []
    for bci in range(len(bin_centers)):
        if not counts[bci]:
            continue
        idx = np.where(bin_idx == bci)[0]
        np.random.seed(37)
        np.random.shuffle(idx)
        total_particle_idxs.append(idx[:num_of_picks])
        total_particle_idxs_eval.append(idx[num_of_picks:num_of_picks+num_of_picks_eval])

    t_e_idxs = {'train':total_particle_idxs, 'eval':total_particle_idxs_eval}

    for k,v in t_e_idxs.items():
        # Concatenate and sort once
        v = np.sort(np.concatenate(v))

        # FAST selection using vectorized lookup
        df_selected = train_particles.iloc[v].copy()

        # Add bin columns (vectorized)
        df_selected["BinIndex"]  = bin_idx[v]
        df_selected["BinCenter"] = bin_centers[df_selected["BinIndex"]]
        df_selected["AnnotationValue"] = train_particles[ann].iloc[v].values

        # Reorder columns
        df_selected = df_selected[[
            "FileIndex", "FileName", "FileParticleIndex",
            "BinIndex", "BinCenter", "AnnotationValue"
        ]]

        if k == 'train':
            ### Important
            df_selected['BinIndex'] = np.unique(df_selected['BinIndex'], return_inverse=True)[1]
            bincenter_binindex_dict = dict(
                zip(df_selected['BinCenter'], df_selected['BinIndex'])
            )
            ### Important

        else:
            df_selected['BinIndex'] = df_selected['BinCenter'].map(bincenter_binindex_dict)
            df_selected = df_selected.dropna(subset=['BinIndex'])
            df_selected['BinIndex'] = df_selected['BinIndex'].astype(int)

        classes, class_counts = np.unique(df_selected['BinIndex'], return_counts=True)
        df_selected.to_parquet(f"./metadata/parquet_files/{k}/{ann}_ue_train.parquet", index=False)


# In[5]:


### Creates particle sets for linear probing training and validation ###
### from the DINOv2 pretraining set, and the Dinov2 validation set, respectively.W###

annotations = ['EMD_Resolution','ReconstructionResolution','MolecularWeight_kDa',\
                'ImagePixelSize','ClassResolution','ClassECA','2DAlignmentClassESS',\
                'DefocusAngle', 'DefocusU', 'DefocusV', 'ProteinClass', 'Symmetry']
#annotations = ['FileName',]
bin_dict = {'MolecularWeight_kDa':[20,0,1000],'DefocusU':[10000,-5000,40000],'DefocusV':[10000,-5000,40000],\
            'DefocusAngle':[30,-105, 105], '2DAlignmentClassESS':[1,0.5,5], 'ClassECA':[0.5,0.75,3],\
            'ClassResolution':[3,1.5,26], 'EMD_Resolution':[0.5,1.25,8], 'ReconstructionResolution':[0.5,1.25,8],\
            'ImagePixelSize':[0.25,0.875,3]}   # store results

for ann in annotations:
    #ann = 'DefocusV'
    # --- detect numeric or non-numeric ---
    train_particles = df_ann.loc[train_idxs]
    if ann in ('ProteinClass', 'Symmetry', 'FileName'):
        np_ann = train_particles[ann]
        bin_centers_, bin_idx, counts = np.unique(np_ann, return_inverse=True, return_counts=True)
        bin_centers = np.arange(len(bin_centers_))
        bin_centers_dict = {c:i for i,c in enumerate(bin_centers_)}
    else:
        if ann == 'ImagePixelSize':
            np_ann = train_particles['ImagePixelSize']*train_particles['ImageSize']/224
        else:
            np_ann = train_particles[ann].astype(float)

        min_v, max_v = np_ann.min(), np_ann.max()        

        if ann in bin_dict:
            bin_dist, start, end = bin_dict[ann]
            bin_edges = np.arange(start, end + bin_dist, bin_dist)
        else:
            bin_dist = 0.5

            # --- compute bin edges only if numeric and not constant ---
            if max_v > min_v:
                start = (np.floor(min_v / bin_dist) * bin_dist)
                end   = ((np.ceil(max_v / bin_dist) * bin_dist))
                start, end = 1.25, 8
                bin_edges = np.arange(start, end + bin_dist, bin_dist)
            else:
                bin_edges = np.array([min_v])  # trivial case
                # Compute bin centers (median of each bin range)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Digitize: assign each sample to a bin index
        bin_idx = np.digitize(np_ann, bin_edges, right=True) - 1
        bin_idx[np.where(bin_idx==-1)] = bin_idx.max()
        #bin_idx = np.clip(bin_idx, 0, len(bin_edges)-2)
        # count per bin (should be roughly equal)
        counts = np.bincount(bin_idx)#, minlength=len(bin_edges)-1)

    num_of_picks_dict = {'ReconstructionResolution':67_000,'EMD_Resolution':60_000,'MolecularWeight_kDa':20_000,\
                     'ImagePixelSize':90_000,'ClassResolution':90_000,'ClassECA':160_000,'2DAlignmentClassESS':160_000,\
                     'DefocusAngle':115_000,'DefocusU':160_000,'DefocusV':160_000,\
                     'ProteinClass':60_000, 'Symmetry':80_000, 'FileName':3_000}#15_000}#3_000}

    np.random.seed(44)
    num_of_picks = num_of_picks_dict[ann]  #(250_000, 350_000, 200_000, 200_000, 350_000, 300_000)

    # Collect indices for all bins
    total_particle_idxs = []
    for bci in range(len(bin_centers)):
        if not counts[bci]:
            continue
        idx = np.where(bin_idx == bci)[0]
        np.random.seed(37)
        np.random.shuffle(idx)
        total_particle_idxs.append(idx[:num_of_picks])

    # Concatenate and sort once
    total_particle_idxs = np.sort(np.concatenate(total_particle_idxs))

    # FAST selection using vectorized lookup
    df_selected = train_particles.iloc[total_particle_idxs].copy()

    # Add bin columns (vectorized)
    df_selected["BinIndex"]  = bin_idx[total_particle_idxs]
    df_selected["BinCenter"] = bin_centers[df_selected["BinIndex"]]
    df_selected["AnnotationValue"] = train_particles[ann].iloc[total_particle_idxs].values

    # Reorder columns
    df_selected = df_selected[[
        "FileIndex", "FileName", "FileParticleIndex",
        "BinIndex", "BinCenter", "AnnotationValue"
    ]]

    ### Important
    df_selected['BinIndex'] = np.unique(df_selected['BinIndex'], return_inverse=True)[1]
    bincenter_binindex_dict = dict(
        zip(df_selected['BinCenter'], df_selected['BinIndex'])
    )
    ### Important

    classes, class_counts = np.unique(df_selected['BinIndex'], return_counts=True)
    #uniq_bin_idxs, bin_idxs_inverse = np.unique(df_data_w_anns['BinIndex'],return_inverse=True)
    df_selected.to_parquet(f"./metadata/parquet_files/train/{ann}_ue_train_on_eval.parquet", index=False)


    ####### EVALUATION #######


    # --- detect numeric or non-numeric ---
    eval_particles = df_ann.loc[eval_idxs]

    if ann in ('ProteinClass', 'Symmetry', 'FileName'):
        eval_cats = eval_particles[ann]#.astype(str)
        bin_idx = np.array(eval_cats.map(bin_centers_dict))
        bin_centers = np.arange(np.max(list(bin_centers_dict.values()))+1)

    else:
        if ann == 'ImagePixelSize':
            np_ann = eval_particles['ImagePixelSize']*eval_particles['ImageSize']/224
        else:
            np_ann = eval_particles[ann].astype(float)

        min_v, max_v = np_ann.min(), np_ann.max()

        if ann in bin_dict:
            bin_dist, start, end = bin_dict[ann]
            bin_edges = np.arange(start, end + bin_dist, bin_dist)
        else:
            bin_dist = 0.5

            # --- compute bin edges only if numeric and not constant ---
            if max_v > min_v:
                start = (np.floor(min_v / bin_dist) * bin_dist)
                end   = ((np.ceil(max_v / bin_dist) * bin_dist))
                start, end = 1.25, 8
                bin_edges = np.arange(start, end + bin_dist, bin_dist)
            else:
                bin_edges = np.array([min_v])  # trivial case
                # Compute bin centers (median of each bin range)

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Digitize: assign each sample to a bin index
        bin_idx = np.digitize(np_ann, bin_edges, right=True) - 1
        bin_idx[np.where(bin_idx==-1)] = bin_idx.max()
        #bin_idx = np.clip(bin_idx, 0, len(bin_edges)-2)
        # count per bin (should be roughly equal)
        counts = np.bincount(bin_idx)#, minlength=len(bin_edges)-1)

    np.random.seed(44)
    num_of_picks = int(0.2*(num_of_picks_dict[ann]))  #(250_000, 350_000, 200_000, 200_000, 350_000, 300_000)

    # Collect indices for all bins
    total_particle_idxs = []
    for bci in range(len(bin_centers)):
        if not counts[bci]:
            continue
        idx = np.where(bin_idx == bci)[0]
        np.random.seed(37)
        np.random.shuffle(idx)
        total_particle_idxs.append(idx[:num_of_picks])

    # Concatenate and sort once
    total_particle_idxs = np.sort(np.concatenate(total_particle_idxs))

    # FAST selection using vectorized lookup
    df_selected = eval_particles.iloc[total_particle_idxs].copy()

    # Add bin columns (vectorized)
    df_selected["BinIndex"]  = bin_idx[total_particle_idxs]
    df_selected["BinCenter"] = bin_centers[df_selected["BinIndex"]]
    df_selected["AnnotationValue"] = eval_particles[ann].iloc[total_particle_idxs].values

    # Reorder columns
    df_selected = df_selected[[
        "FileIndex", "FileName", "FileParticleIndex",
        "BinIndex", "BinCenter", "AnnotationValue"
    ]]

    ###Important map the correct class indexes
    df_selected['BinIndex'] = df_selected['BinCenter'].map(
    bincenter_binindex_dict)

    df_selected = df_selected.dropna(subset=['BinIndex'])

    df_selected['BinIndex'] = df_selected['BinIndex'].astype(int)
    ###Important map the correct class indexes

    classes, class_counts = np.unique(df_selected['BinIndex'], return_counts=True)
    #uniq_bin_idxs, bin_idxs_inverse = np.unique(df_data_w_anns['BinIndex'],return_inverse=True)
    df_selected.to_parquet(f"./metadata/parquet_files/eval/{ann}_ue_train_on_eval.parquet", index=False)

