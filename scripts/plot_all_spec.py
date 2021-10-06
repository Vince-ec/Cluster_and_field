import numpy as np
import pandas as pd
from shutil import copyfile
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import fsps
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
# from sim_engine import Scale_model
# from spec_tools import Source_present, Oldest_galaxy, Sig_int, Smooth, Rescale_SF_sfh, Posterior_SF_spec, lbt_to_z
# from spec_stats import Smooth, Highest_density_region
# from spec_id import *
from spec_stats import Highest_density_region, Linear_fit
from spec_exam import Gen_SF_spec, Gen_spec, Gen_spec_2D
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as seadddd
import os
from grizli import multifit
from grizli import model
# from sim_engine import forward_model_grism

import seaborn as sea
from time import time
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)

NGSID = np.load('../dataframes/N_GSD_2.npy',  allow_pickle=True)
NGNID = np.load('../dataframes/N_GND_2.npy',  allow_pickle=True)
NGSz = np.load('../dataframes/N_GSD_2_z.npy', allow_pickle=True)
NGNz = np.load('../dataframes/N_GND_2_z.npy', allow_pickle=True)
print(len(NGSID))
print(len(NGNID))

# field='GSD'
# pre = field[1]

# for i in range(len(NGSID)):
#     gid  = NGSID[i]
#     rshift = NGSz[i]

#     if gid < 10000:
#         GID = '0' + str(gid)
#     else:
#         GID = str(gid)

#     Gs = Gen_spec_2D('G{}D'.format(pre), GID, rshift)
    
#     plt.figure(figsize = [15,13])
#     plt.subplot(211)
#     if Gs.g102:
#         plt.errorbar(Gs.Bwv,Gs.Bfl,Gs.Ber,
#                     linestyle='None', marker='o', markersize=3, color='#377eb8',zorder = 2, label = 'CLEAR G102')
#         IDMB = np.repeat(True, len(Gs.Bwv))

#     if Gs.g141:
#         plt.errorbar(Gs.Rwv,Gs.Rfl,Gs.Rer,
#                     linestyle='None', marker='o', markersize=3, color='#e41a1c',zorder = 2, label = '3D-HST G141')
#         IDMR = np.repeat(True, len(Gs.Rwv))

#     plt.errorbar(Gs.Pwv,Gs.Pflx,Gs.Perr,
#                     linestyle='None', marker='o', markersize=10, color='#4daf4a',zorder = 1, label = '3D-HST Photometry')

#     plt.axvline(2799.177 * (1 + rshift),linestyle='--', alpha=.3) # MGII
#     plt.axvline(3727.092 * (1 + rshift),linestyle='--', alpha=.3) # OII
#     plt.axvline(4102.89 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(4341.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(4862.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(5008.240 * (1 + rshift),linestyle='--', alpha=.3)
#     plt.axvline(6564.61 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(6718.29 * (1 + rshift),linestyle='--', alpha=.3, color = 'k')

#     plt.xlabel('Wavelength ($\AA$)', fontsize=25)
#     plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=25)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     plt.xlim(7500,16500)
#     plt.title('{}-{}'.format(field, gid))

#     plt.subplot(212)
#     if Gs.g102:
#         plt.errorbar(Gs.Bwv,Gs.Bfl,Gs.Ber,
#                 linestyle='None', marker='o', markersize=3, color='#377eb8',zorder = 2, label = 'CLEAR G102')
#     if Gs.g141:
#         plt.errorbar(Gs.Rwv,Gs.Rfl,Gs.Rer,
#                     linestyle='None', marker='o', markersize=3, color='#e41a1c',zorder = 2, label = '3D-HST G141')
#     plt.errorbar(Gs.Pwv,Gs.Pflx,Gs.Perr,
#                     linestyle='None', marker='o', markersize=10, color='#4daf4a',zorder = 1, label = '3D-HST Photometry')


#     plt.axvline(3727.092 * (1 + rshift),linestyle='--', alpha=.3)
#     plt.axvline(4102.89 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(4341.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(4862.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(5008.240 * (1 + rshift),linestyle='--', alpha=.3)
#     plt.axvline(6564.61 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
#     plt.axvline(6718.29 * (1 + rshift),linestyle='--', alpha=.3, color = 'k')

#     plt.xlabel('Wavelength ($\AA$)', fontsize=25)
#     plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=25)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     plt.xscale('log')
#     plt.savefig('../plots/newspec_check/{}-{}_rshift_check.png'.format(field, gid), bbox_inches = 'tight')    
    
    
field='GND'
pre = field[1]

for i in range(len(NGNID)):
    gid  = NGNID[i]
    rshift = NGNz[i]

    if gid < 10000:
        GID = '0' + str(gid)
    else:
        GID = str(gid)

    Gs = Gen_spec_2D('G{}D'.format(pre), gid, rshift)
    
    plt.figure(figsize = [15,13])
    plt.subplot(211)
    if Gs.g102:
        plt.errorbar(Gs.Bwv,Gs.Bfl,Gs.Ber,
                    linestyle='None', marker='o', markersize=3, color='#377eb8',zorder = 2, label = 'CLEAR G102')
        IDMB = np.repeat(True, len(Gs.Bwv))

    if Gs.g141:
        plt.errorbar(Gs.Rwv,Gs.Rfl,Gs.Rer,
                    linestyle='None', marker='o', markersize=3, color='#e41a1c',zorder = 2, label = '3D-HST G141')
        IDMR = np.repeat(True, len(Gs.Rwv))

    plt.errorbar(Gs.Pwv,Gs.Pflx,Gs.Perr,
                    linestyle='None', marker='o', markersize=10, color='#4daf4a',zorder = 1, label = '3D-HST Photometry')

    plt.axvline(2799.177 * (1 + rshift),linestyle='--', alpha=.3) # MGII
    plt.axvline(3727.092 * (1 + rshift),linestyle='--', alpha=.3) # OII
    plt.axvline(4102.89 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4341.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4862.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(5008.240 * (1 + rshift),linestyle='--', alpha=.3)
    plt.axvline(6564.61 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(6718.29 * (1 + rshift),linestyle='--', alpha=.3, color = 'k')

    plt.xlabel('Wavelength ($\AA$)', fontsize=25)
    plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlim(7500,16500)
    plt.title('{}-{}'.format(field, gid))

    plt.subplot(212)
    if Gs.g102:
        plt.errorbar(Gs.Bwv,Gs.Bfl,Gs.Ber,
                linestyle='None', marker='o', markersize=3, color='#377eb8',zorder = 2, label = 'CLEAR G102')
    if Gs.g141:
        plt.errorbar(Gs.Rwv,Gs.Rfl,Gs.Rer,
                    linestyle='None', marker='o', markersize=3, color='#e41a1c',zorder = 2, label = '3D-HST G141')
    plt.errorbar(Gs.Pwv,Gs.Pflx,Gs.Perr,
                    linestyle='None', marker='o', markersize=10, color='#4daf4a',zorder = 1, label = '3D-HST Photometry')


    plt.axvline(3727.092 * (1 + rshift),linestyle='--', alpha=.3)
    plt.axvline(4102.89 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4341.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(4862.68 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(5008.240 * (1 + rshift),linestyle='--', alpha=.3)
    plt.axvline(6564.61 * (1 + rshift),linestyle='--', alpha=.3, color = 'r')
    plt.axvline(6718.29 * (1 + rshift),linestyle='--', alpha=.3, color = 'k')

    plt.xlabel('Wavelength ($\AA$)', fontsize=25)
    plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xscale('log')
    plt.savefig('../plots/newspec_check/{}-{}_rshift_check.png'.format(field, gid), bbox_inches = 'tight')   