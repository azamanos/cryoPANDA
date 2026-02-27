#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import pandas as pd
from utils.utils import box_sizes
from utils.scraping_utils import fetch_empiar_emd_metadata_and_pdb_data_multiprocess


# In[2]:


# ---------------------------------------------------------------------------
# EMPIAR ID lists
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


# In[3]:


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


# In[4]:


# ---------------------------------------------------------------------------
# Build EMPIAR -> [EMD] mapping
# ---------------------------------------------------------------------------

empiar_emd_dict = {}

for entry in csv_info[1:]:
    empiar = entry[0].split('-')[-1]

    if empiar in total_entries:
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


# In[5]:


# ---------------------------------------------------------------------------
# Read selected EMPIAR/EMD pairs from Excel
# ---------------------------------------------------------------------------

file_path = './metadata/All_Jobs.xlsx'
df = pd.read_excel(file_path)

selected_empiar_emd = np.concatenate(
    (np.array(df['EMPIAR_ID'])[:, None], np.array(df['EMD_ID'])[:, None]),
    axis=1,
)

selected_empiar_emd = np.array(
    [
        [str(i[0]).split('_')[0], str(i[1])]
        for i in selected_empiar_emd
        if str(i[0]).split('_')[0] != 'Kastritis'
    ]
)

selected_empiar_emd[
    np.where(selected_empiar_emd[:, 0] == '10407')[0], 1
] = '10898'

info_dict = {}

for i in info:
    empiar_id = list(i.keys())[0]
    info_dict[empiar_id] = list(i.values())[0]
    if empiar_id in selected_empiar_emd:
        selected_emd = str(selected_empiar_emd[np.where(selected_empiar_emd[:,0]==empiar_id)[0],1][0])
        info_dict[empiar_id]['emd_info'] = {selected_emd:info_dict[empiar_id]['emd_info'][selected_emd]}


# In[6]:


# ---------------------------------------------------------------------------
# Manual corrections / overrides
# ---------------------------------------------------------------------------

change_reported_box = {'11847': 256, '11116': 224, '10199': 600}

dn, up, box_mult = 1, 1, 2

change_reported_min_max = {'11898':[420*dn,420*up], '11341':[50*dn,140*up], '11405':[240*dn,340*up], '11443':[60*dn,110*up],\
                           '11485':[80*dn,100*up],'11564':[100*dn,115*up], '11567':[60*dn,100*up], '11556':[90*dn,90*up],\
                           '12140':[90*dn,170*up],'10017':[140*dn,175*up],'10025':[115*dn,160*up],'10096':[70*dn,120*up],\
                           '10097':[70*dn,135*up],'10122':[130*dn,130*up],'10175':[75*dn,130*up],'10189':[1300*dn,1300*up],\
                           '10198':[330*dn,330*up],'10217':[100*dn,140*up],'10271':[180*dn,200*up],'10373':[70*dn,170*up],\
                           '10379':[100*dn,105*up],'10590':[170*dn,200*up],'10704':[70*dn,190*up],'10800':[75*dn,120*up],\
                           '10801':[105*dn,130*up],'10888':[850*dn,850*up],'10890':[750*dn,750*up],'11024':[200*dn,230*up],\
                           '11077':[300*dn,340*up],'10241':[130*dn,300*up],'12240':[120*dn,120*up],\
                           '12237':[200*dn,305*up], '11791':[260*dn,270*up],'11792':[90*dn,90*up], '11796':[130*dn,130*up],\
                           '11693':[125*dn,420*up],'10876':[80*dn,170*up], '10268':[200*dn,260*up], '10215':[120*dn,160*up],\
                           '10075':[275*dn,285*up],'10160':[290*dn,320*up],'11034':[300*dn,340*up],'11354':[120*dn,165*up],\
                           '10652':[300*dn,300*up],'11034':[310*dn,380*up],'10005':[100*dn,115*up],'10193':[480*dn,500*up],\
                           '11194':[55*dn,95*up], '11681':[70*dn,115*up], '11834':[120*dn,175*up], '11618':[100*dn,130*up],\
                           '11120':[80*dn,180*up], '10841':[200*dn,220*up], '10752':[120*dn,215*up], '10128':[100*dn,125*up],\
                           '10420':[80*dn,115*up], '10455':[850*dn,850*up], '10335':[45*dn,55*up], '10294':[750*dn,750*up],\
                           '10229':[55*dn,85*up], '10099':[120*dn,140*up], '10076':[200*dn,205*up]}

change_symmetry = {'11768':'D2', '11501':'C2', '10652':'I', '10075':'I', '10799':'C2', '11861':'C2', '12171':'C2', '10005':'C4',\
                   '10012':'D2', '10025':'D7', '10189':'I', '10193':'I', '10203':'I', '10240':'C2', '10425':'C2', '10760':'C7',\
                   '10801':'C2', '10893':'C13','11149':'C2', '11719':'C2','12240':'O', '10790':'C6'}
change_acceleration_voltage_kV = {'10652':300}
change_nominal_cs_mm = {'11986':2.7, '12562':2.7, '11847':2.7, '12140':2.7, '11755':2.7, '11556':2.7,\
                        '11763':2.7, '12106':2.7, '12087':2.7, '11898':2.7, '11501':2.7, '10735':2.7,\
                        '10652':2.7, '10075':2.7, '10203':2.7, '10379':2.0}
change_micrograph_pixel_size = {'11443':0.36}
change_electron_dose_per_image = {'11768':22.5, '10652':47, '10429':92}

no_ps_dict = {'11986': 0.88, '11755': 1.069, '12106': 0.85, '12087': 1.094, '11726': 1.083, '11341': 0.826, '10707': 0.85,\
              '10706': 1.0277, '10704': 1.35, '10667': 1.35, '10291': 1.232, '10289': 1.232, '10290': 1.232, '10285': 1.0,\
              '10241': 1.096, '10240': 1.096, '10239': 1.096, '11906': 0.74, '11608':0.835, '10937':1.71, '11120':0.83,\
              '10483': 1.07, '10299': 1.40}


# ---------------------------------------------------------------------------
# Build CryoSPARC metadata dict per EMPIAR
# ---------------------------------------------------------------------------

cryosparc_dict = {}
test_box_sizes = []

keys_to_extract = [
    'acceleration_voltage_kV',
    'nominal_cs_mm',
    'electron_dose_per_image',
]

micrograph_empiars = [me for me in info_dict.keys()]

for i in micrograph_empiars:
    temp_cryosparc_dict = {}

    temp_emp_dict_key = list(info_dict[i]['empiar_info'].keys())[0]
    temp_emp_dict = info_dict[i]['empiar_info'][temp_emp_dict_key]
    temp_emd_dict = dict(list(info_dict[i]['emd_info'].values())[0])

    # -----------------------------------------------------------------------
    # Pixel size
    # -----------------------------------------------------------------------
    emd_pixel_spacing = np.array(
        list(temp_emd_dict['pixel_spacing'].values())
    ).astype(float)
    all_equal = np.all(emd_pixel_spacing == emd_pixel_spacing[0])

    if temp_emp_dict['pixelWidth'] == temp_emp_dict['pixelHeight']:
        try:
            temp_cryosparc_dict['micrographs_pixel_size'] = float(
                temp_emp_dict['pixelWidth']
            )
        except Exception:
            temp_cryosparc_dict['micrographs_pixel_size'] = no_ps_dict[i]

    elif temp_emp_dict['pixelWidth'] == 'variable':
        temp_cryosparc_dict['micrographs_pixel_size'] = float(
            temp_emp_dict['pixelHeight']
        )

    elif temp_emp_dict['pixelHeight'] == 'variable':
        temp_cryosparc_dict['micrographs_pixel_size'] = float(
            temp_emp_dict['pixelWidth']
        )

    else:
        print(i)
        temp_cryosparc_dict['authors_map_pixel_size'] = float(
            emd_pixel_spacing[0]
        )

    if all_equal:
        temp_cryosparc_dict['authors_map_pixel_size'] = float(
            emd_pixel_spacing[0]
        )
    else:
        print(f"Pixels Spacing Problem in Authors Map EMPIAR {i}")

    if i in change_micrograph_pixel_size:
        temp_cryosparc_dict['micrographs_pixel_size'] = (
            change_micrograph_pixel_size[i]
        )

    # -----------------------------------------------------------------------
    # Beam / dose metadata: 'acceleration_voltage_kV', 'nominal_cs_mm',
    # 'electron_dose_per_image'
    # -----------------------------------------------------------------------
    temp_cryosparc_dict = temp_cryosparc_dict | {
        k: temp_emd_dict[k] for k in keys_to_extract if k in temp_emd_dict
    }

    if i in change_acceleration_voltage_kV:
        temp_cryosparc_dict['acceleration_voltage_kV'] = str(
            change_acceleration_voltage_kV[i]
        )
    if i in change_nominal_cs_mm:
        temp_cryosparc_dict['nominal_cs_mm'] = str(
            change_nominal_cs_mm[i]
        )
    if i in change_electron_dose_per_image:
        temp_cryosparc_dict['electron_dose_per_image'] = str(
            change_electron_dose_per_image[i]
        )

    # -----------------------------------------------------------------------
    # Particle diameter / box size
    # -----------------------------------------------------------------------
    temp_min, temp_max = 1_000_000, 0

    for mm in list(info_dict[i]['pdb_info'].values()):
        if mm['min_diameter'] < temp_min:
            temp_min = mm['min_diameter']
        if mm['max_diameter'] > temp_max:
            temp_max = mm['max_diameter']

    if i in ('11405', '10271'):
        temp_min, temp_max = 1_000_000, 0

    if temp_min == 1_000_000 and temp_max == 0:
        temp_cryosparc_dict['min_diameter'] = 0
        temp_cryosparc_dict['max_diameter'] = 0
    else:
        temp_min_, temp_max_ = int(temp_min * dn), int(temp_max * up)
        temp_cryosparc_dict['min_diameter'] = temp_min_
        temp_cryosparc_dict['max_diameter'] = temp_max_

    if i in change_reported_min_max:
        temp_cryosparc_dict['min_diameter'] = change_reported_min_max[i][0]
        temp_cryosparc_dict['max_diameter'] = change_reported_min_max[i][1]

    emd_box_size = int(
        np.max(np.array(list(temp_emd_dict['dimensions'].values()), dtype=int))
    )

    try:
        if temp_cryosparc_dict['max_diameter']:
            min_box_pdb_box_size = (
                box_mult
                * (
                    (temp_cryosparc_dict['max_diameter'] / up)
                    / float(temp_cryosparc_dict['micrographs_pixel_size'])
                )
            )
            min_box_pdb_box_size = int(
                box_sizes[np.where(box_sizes / min_box_pdb_box_size > 1)[0][0]]
            )
        else:
            min_box_pdb_box_size = 0
    except Exception:
        min_box_pdb_box_size = 0

    temp_cryosparc_dict['authors_extraction_box_size'] = emd_box_size

    if i in change_reported_box:
        temp_cryosparc_dict['authors_extraction_box_size'] = (
            change_reported_box[i]
        )

    test_box_sizes.append(
        [
            i,
            temp_cryosparc_dict['authors_extraction_box_size'],
            min_box_pdb_box_size,
            emd_pixel_spacing[0],
            temp_cryosparc_dict['micrographs_pixel_size'],
        ]
    )

    if not min_box_pdb_box_size:
        # If pixel sizes disagree, rescale the box
        if (
            temp_cryosparc_dict['micrographs_pixel_size']
            and abs(
                temp_cryosparc_dict['micrographs_pixel_size']
                - temp_cryosparc_dict['authors_map_pixel_size']
            )
            > 0.1
        ):
            min_box_pdb_box_size = emd_box_size / (
                temp_cryosparc_dict['micrographs_pixel_size']
                / temp_cryosparc_dict['authors_map_pixel_size']
            )
            if min_box_pdb_box_size < 4096:
                min_box_pdb_box_size = int(
                    box_sizes[
                        np.where(box_sizes / min_box_pdb_box_size > 1)[0][0]
                    ]
                )
            else:
                min_box_pdb_box_size = 'too large'

    temp_cryosparc_dict['computed_extraction_box_size'] = (
        min_box_pdb_box_size
    )

    temp_cryosparc_dict['symmetry'] = temp_emd_dict['symmetry']
    if i in change_symmetry:
        temp_cryosparc_dict['symmetry'] = change_symmetry[i]

    cryosparc_dict[i] = temp_cryosparc_dict

test_box_sizes = np.round(np.array(test_box_sizes).astype(float), 2)

cryosparc_df = pd.DataFrame.from_dict(cryosparc_dict, orient='index')


# In[7]:


### Extracted info are contained within cryosparc_df ###
cryosparc_df


# In[8]:


cryosparc_df['EMPIAR_ID'] = cryosparc_df.index
cryosparc_df.to_parquet("./metadata/EMPIAR_statistics.parquet", index=False)

