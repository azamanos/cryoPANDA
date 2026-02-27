#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import h5py
import numpy as np
import pandas as pd


# In[2]:


extract_columns = ['ReconstructionResolution', 'EMD_Resolution', 'MolecularWeight_kDa',
 'ImagePixelSize', 'ImageSize', 'ClassResolution', 'ClassECA',
 '2DAlignmentClassESS', 'DefocusAngle', 'DefocusU', 'DefocusV',
 'ProteinClass', 'Symmetry', 'ClassNumber',
 'AngleRot', 'AngleTilt', 'AnglePsi',
 'Subunits', 'MaxDiameter', 'DNA', 'RNA']

path = 'PATH_TO_DIRECTORY_WITH_H5_FILES'
h5_list = sorted(os.listdir(path))[:]
lb = 0
concatenate_files = []
for ii, i in enumerate(np.linspace(50,252,5, dtype=int)):
    for file_index, h5_file in enumerate(h5_list[lb:i]):
        with h5py.File(path+h5_file, "r") as f:
            annotations_json = f["annotations"][()].decode()
            annotations_dict = json.loads(annotations_json)
            df_exp = pd.DataFrame.from_dict(annotations_dict)
        if h5_file == '10407_particles.h5':
            df_exp['EMD_Resolution'] = 2.7
        df_exp_ = df_exp[extract_columns].fillna(df_exp.loc[df_exp.index[0], extract_columns])
        df_exp_.insert(0, 'FileIndex', file_index+lb)
        df_exp_.insert(1, 'FileName', h5_file)
        df_exp_.insert(2, 'FileParticleIndex', np.arange(len(df_exp_)))

        if not file_index:
            df_ann_ = df_exp_
        else:
            df_ann_ = pd.concat([df_ann_, df_exp_], ignore_index=True)
        print(f'{file_index+1+lb} {h5_file}', end='\r')
    lb = i
    fp = f"./metadata/parquet_files/df_ann_{i}.parquet"
    concatenate_files.append(fp)
    df_ann_.to_parquet(fp)
df_ann_ = pd.concat([pd.read_parquet(concatenate_files[0]), pd.read_parquet(concatenate_files[1]),\
                     pd.read_parquet(concatenate_files[2]), pd.read_parquet(concatenate_files[3]),\
                     pd.read_parquet(concatenate_files[4])], ignore_index=True)
df_ann_.to_parquet(f"./metadata/parquet_files/df_ann.parquet")
[os.remove(i) for i in concatenate_files]

