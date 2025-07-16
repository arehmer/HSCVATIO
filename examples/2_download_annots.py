# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:50:25 2023

@author: Rehmer
"""

from pathlib import Path
from hsod.od_mgr import od_mgr


project_name = 'Archesens_32x32_L1k9_ang90'

# %% Specify where the data that should be imported is located
hdf5_path = Path.cwd() /  (project_name+'.hdf5')

# %%  Initialize hdf5 manager
data_mgr = od_mgr(hdf5_path.as_posix(),mode='a')

# %% Sync hdf5 with cvat server
data_mgr.sync_with_cvat(project_name)
hdf5_index = data_mgr.load_index()
    
# %% Find existing annotations Download annotations
download_idx = [i for i in hdf5_index.index if \
                (hdf5_index.loc[i,'stage'] == 'validation') |\
                (hdf5_index.loc[i,'stage'] == 'acceptance')   ]


# %% Download annotations
for i in download_idx:
    
    data_mgr.cvat_download_annotation([i],mode='w')
    
    
