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
from spec_id_2d import args
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

############
def edit_img(fl, ID):
    mb = multifit.MultiBeam(fl,**args)

    ### step 1 isolate
    iso = np.array(mb.beams[0].beam.direct*(mb.beams[0].beam.seg == ID))


    ### step 2 zoom in
    ziso = iso[59:99,59:99]


    ### step 3 get 3% and 97% - tiles
    llim = np.percentile(ziso[ziso != 0], 3)
    hlim = np.percentile(ziso[ziso != 0], 97)
    
    ### step 4 cap lower and under values
    ziso[ziso < llim] = llim
    ziso[ziso > hlim] = hlim
    
    hlim -= llim

    ### step 5 set lower to 0 

    for i in range(len(ziso)):
        for ii in range(len(ziso[0])):
            if ziso[i][ii] != 0:
                ziso[i][ii] -= llim

    ### step 6 set top to 1
    ziso /= hlim
    return ziso
############

for i in Adb.index:
    field = Adb.field[i]
    gid = Adb.id[i]
    fl = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/BEAMS/*{}*/*{}*'.format(field[1], gid))[0]
    
    ziso = edit_img(fl, gid)
    
    np.save('../plots/evolution_plots/morph_gif/{}-{}'.format(field, gid), ziso)
