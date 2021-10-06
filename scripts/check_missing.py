import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_sfh
from glob import glob
import os
import pickle
from make_sfh_tool import Gen_sim_SFH

### set home for files
hpath = os.environ['HOME'] + '/'

data_path = '/scratch/user/vestrada78840/data/'
pos_path = '/home/vestrada78840/posteriors/'
sfh_path = '/scratch/user/vestrada78840/SFH/'

NGSID = np.load(data_path + 'N_GSD_2.npy',  allow_pickle=True)
NGNID = np.load(data_path + 'N_GND_2.npy',  allow_pickle=True)

NGSsf = np.load(data_path + 'N_GSD_2_sf.npy',  allow_pickle=True)
NGNsf = np.load(data_path + 'N_GND_2_sf.npy',  allow_pickle=True)


###### create tabit db#########
idx = 1
for i in range(len(NGSID)):
    if NGSsf[i] == 'Q':
        if not os.path.isfile(sfh_path + 'GSD_{}_1D.pkl'.format(NGSID[i])):
            print('GSD-{}'.format(NGSID[i]))
    else:
        if not os.path.isfile(sfh_path + 'GSD_{}_p1_1D.pkl'.format(NGSID[i])):
            print('GSD-{}'.format(NGSID[i]))
    idx+=1
    
for i in range(len(NGNID)):
    if NGNsf[i] == 'Q':
        if not os.path.isfile(sfh_path + 'GND_{}_1D.pkl'.format(NGNID[i])):
            print('GND-{}'.format(NGNID[i]))
    else:
        if not os.path.isfile(sfh_path + 'GND_{}_p1_1D.pkl'.format(NGNID[i])):
            print('GND-{}'.format(NGNID[i]))
    idx+=1
