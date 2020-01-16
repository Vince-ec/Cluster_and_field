import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.interpolate import interp1d, interp2d

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sea

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from sim_engine import Scale_model
from spec_tools import Source_present, Oldest_galaxy, Sig_int, Smooth
from spec_stats import Smooth, Highest_density_region
from spec_id import *
from spec_exam import Gen_spec

from grizli import multifit
from grizli import model
from grizli.utils import SpectrumTemplate


sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)
### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/Users/Vince.ec/':
    dpath = '/Volumes/Vince_research/Data/' 
    
else:
    dpath = hpath + 'Data/' 

########################################################################################
########################################################################################
    
Sids = [23102,24622,25053,25884,27965,29928,40985,42985,44471,46275,42113,43114,43683,42607,44133]
Sz =   [0.664,0.680,0.631,0.666,0.619,1.090,0.748,0.733,0.732,0.717,1.615,1.890,1.888,2.269,2.140]
Nids = [12006,17194,20538,22633,23857,38225,33777,15976,26197,38061]
Nz =   [0.664,1.019,0.641,1.014,1.120,1.226,1.769,2.493,2.053,1.815]
        
pre = 'N'

for i in range(len(Nids)):
    gid = Nids[i]
    rshift = Nz[i]
    Gs = Gen_spec('G{}D'.format(pre), gid, rshift,decontam = False)
    
    plt.figure(figsize = [16,8])
    
    ax2 = plt.subplot()

    if Gs.g102:
        ax2.errorbar(Gs.Bwv_rf,Gs.Bfl *1E18,Gs.Ber *1E18,
                linestyle='None', marker='o', markersize=0.25, color='#36787A', zorder = 2, elinewidth = 0.4)

    if Gs.g141:
        ax2.errorbar(Gs.Rwv_rf,Gs.Rfl *1E18,Gs.Rer *1E18,
                linestyle='None', marker='o', markersize=0.25, color='#EA2E3B', zorder = 2, elinewidth = 0.4)

    plt.axvline(3727.092,linestyle='--', alpha=.3) # OII
    plt.axvline(4102.89,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4341.68,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4862.68,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(5008.240,linestyle='--', alpha=.3)
    plt.axvline(6564.61,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(6718.2,linestyle='--', alpha=.3, color = 'k')
    plt.savefig('../plots/newspec_exam/G{}D-{}_beams.png'.format(pre, gid),bbox_inches = 'tight')

    
pre = 'S'

for i in range(len(Sids)):
    gid = Sids[i]
    rshift = Sz[i]
    Gs = Gen_spec('G{}D'.format(pre), gid, rshift,decontam = False)
    
    plt.figure(figsize = [16,8])
    
    ax2 = plt.subplot()

    if Gs.g102:
        ax2.errorbar(Gs.Bwv_rf,Gs.Bfl *1E18,Gs.Ber *1E18,
                linestyle='None', marker='o', markersize=0.25, color='#36787A', zorder = 2, elinewidth = 0.4)

    if Gs.g141:
        ax2.errorbar(Gs.Rwv_rf,Gs.Rfl *1E18,Gs.Rer *1E18,
                linestyle='None', marker='o', markersize=0.25, color='#EA2E3B', zorder = 2, elinewidth = 0.4)

    plt.axvline(3727.092,linestyle='--', alpha=.3) # OII
    plt.axvline(4102.89,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4341.68,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4862.68,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(5008.240,linestyle='--', alpha=.3)
    plt.axvline(6564.61,linestyle='--', alpha=.3, color = 'r')
    plt.axvline(6718.2,linestyle='--', alpha=.3, color = 'k')
    plt.savefig('../plots/newspec_exam/G{}D-{}_beams.png'.format(pre, gid),bbox_inches = 'tight')