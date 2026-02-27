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


train_particles = df_ann.loc[train_idxs]
train_particles["BinIndex"] = 0
train_particles["BinCenter"] = 'train_particle'
train_particles["AnnotationValue"] = 'train_particle'

# Reorder columns
train_particles = train_particles[[
    "FileIndex", "FileName", "FileParticleIndex",
    "BinIndex", "BinCenter", "AnnotationValue"
]]

train_particles.to_parquet(f"./metadata/parquet_files/train_ue.parquet", index=False)

eval_particles = df_ann.loc[eval_idxs]
eval_particles["BinIndex"] = 1
eval_particles["BinCenter"] = 'eval_particle'
eval_particles["AnnotationValue"] = 'eval_particle'

# Reorder columns
eval_particles = eval_particles[[
    "FileIndex", "FileName", "FileParticleIndex",
    "BinIndex", "BinCenter", "AnnotationValue"
]]

eval_particles.to_parquet(f"./metadata/parquet_files/eval_ue.parquet", index=False)

