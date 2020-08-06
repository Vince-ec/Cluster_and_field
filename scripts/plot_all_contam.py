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
from spec_exam import Gen_spec_2D

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
    
GND_all = pd.read_pickle('../dataframes/galaxy_frames/GND_CLEAR.pkl')
GSD_all = pd.read_pickle('../dataframes/galaxy_frames/GSD_CLEAR.pkl')

GND_all = GND_all.query('good_spec == 1')
GSD_all = GSD_all.query('good_spec == 1')

"""
pre = 'N'

for i in GND_all.index:
    gid = GND_all.id[i]
    if gid < 10000:
        gid = '0' + str(gid)
    rshift = GND_all.zgrizli[i][0]
    Gs = Gen_spec_2D('G{}D'.format(pre), gid, rshift)
    
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
"""
    
pre = 'S'

for i in GSD_all.index:
    gid = GSD_all.id[i]
    rshift = GSD_all.zgrizli[i][0]
    if not os.path.isfile('../plots/newspec_exam/G{}D-{}_beams.png'.format(pre, gid)):

        Gs = Gen_spec_2D('G{}D'.format(pre), gid, rshift)

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