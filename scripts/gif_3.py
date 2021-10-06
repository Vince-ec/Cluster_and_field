import numpy as np
import pandas as pd
from shutil import copyfile
# from astropy.cosmology import Planck13 as cosmo
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import fsps
from matplotlib import gridspec
import matplotlib as mpl
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import astropy.units as u
from sim_engine import Scale_model
from spec_tools import Source_present, Oldest_galaxy, Sig_int, Smooth, Rescale_sfh, \
    lbt_to_z, boot_to_posterior, age_to_z, Posterior_spec
from spec_stats import Smooth, Highest_density_region, Linear_fit
from spec_id import *
from spec_id_2d import Gen_temp_dict_addline
from spec_stats import Highest_density_region, Linear_fit
from spec_exam import Gen_spec
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as sea
import os
from grizli import multifit
from grizli import model
from grizli.utils import SpectrumTemplate
from sim_engine import forward_model_grism
from scipy import stats
import pickle
from spec_tools import Gen_SFH
from spec_tools import Photometry
from spec_stats import Iterative_stacking

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})

from matplotlib.colors import ListedColormap
clist = [[166, 58, 0],
[72, 146, 86],
[20, 57, 80]]
X = np.linspace(-15, -7, 500)

A = []
B = []
C = []
for i in range(2):
    A.extend(np.linspace(clist[i][0]/255, clist[i+1][0]/255, 500))
    B.extend(np.linspace(clist[i][1]/255, clist[i+1][1]/255, 500))
    C.extend(np.linspace(clist[i][2]/255, clist[i+1][2]/255, 500))
CMAP = ListedColormap(np.array([A,B,C]).T)



Adb = pd.read_pickle('../dataframes/fitdb/evolution_db_masslim.pkl')
Adb = Adb.query('id != 44707')
pdb = Adb.sort_values('sf_prob_ml', ascending = False)
pdb.reset_index(drop=True, inplace=True)


idx = 0

while len(pdb.index[0 + idx:100 + idx]) == 100:    
    DB = pdb.query('{} < index < {}'.format(0 + idx,100 + idx))

    img = np.zeros([40,40])
    for i in DB.index:
        img += np.load('../plots/evolution_plots/morph_gif/{}-{}.npy'.format(DB.field[i], DB.id[i]))

    
    plt.figure(figsize = [10,10])
    plt.imshow(np.arcsinh(img), vmin = 2)
    plt.xticks([])
    plt.yticks([])

    plt.savefig('../plots/evolution_plots/morph_gif/morph_{}.png'.format(idx), bbox_inches = 'tight')    
  
    idx +=1
    
#     if idx == 245:
#         break