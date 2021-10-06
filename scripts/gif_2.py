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


##########################################################
def Stack(wv, flxgrid , errgrid):
    flgrid = np.transpose(flxgrid)
    errgrid = np.transpose(errgrid)
    weigrid = errgrid ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack, err = np.zeros([2, len(wv)])
    for i in range(len(wv)):
        stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))        
        err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
    ################
    
    return stack, err

def stack_setup(db, swave):
    flxgrid = []
    errgrid = []

    for i in db.index:
        try:
            ffl = np.zeros_like(swave).astype(float)
            fer = np.zeros_like(swave).astype(float)
            sp = np.load('../full_specs/{}_{}_fullspec.npy'.format(db.field[i], db.id[i]), allow_pickle=True).item()
            IDX = [U for U in range(len(sp['wave'])) if 5400 < sp['wave'][U] < 5800]
            Nfact = np.trapz(sp['flam'][IDX],sp['wave'][IDX])
    #         print(Nfact)

            if not Nfact > 0:
                print(db.field[i], db.id[i],i)

            if len(sp['Bwv']) > 0:
                bwv = sp['Bwv'] / (1 + db.zgrism[i])
                bfl = interp1d(bwv,sp['Bfl']/Nfact, fill_value=0, bounds_error=False)(swave)
                ber = interp1d(bwv,sp['Ber']/Nfact, fill_value=0, bounds_error=False)(swave)
                ffl += bfl
                fer += ber

            if len(sp['Rwv']) > 0:
                rwv = sp['Rwv'] / (1 + db.zgrism[i])
                rfl = interp1d(rwv,sp['Rfl']/Nfact, fill_value=0, bounds_error=False)(swave)
                rer = interp1d(rwv,sp['Rer']/Nfact, fill_value=0, bounds_error=False)(swave)
                ffl += rfl
                fer += rer
            fer[fer == 0] = 1
            flxgrid.append(ffl)
            errgrid.append(fer)   
        except:
            print('error',db.field[i], db.id[i],i)
            
    W = Leave_one_out_STACK(swave,flxgrid,errgrid)
    
    for i in range(len(W)):
        errgrid[i] = np.sqrt(errgrid[i]**2 + (flxgrid[i]*W[i])**2)
    
    return Stack(swave, flxgrid, errgrid)#, flxgrid, errgrid

def plot_spec(ax,wv,fl,er, lim1, lim2, color, label):
    IDS = [U for U in range(len(wv)) if lim1 < wv[U] < lim2]
    ax.errorbar(wv[IDS], fl[IDS], er[IDS], fmt = 'o', markersize = 2,color=color,label = '{}'.format(label))
    # O
    ax.axvline(3727.092 ,linestyle='--', alpha=0.3, color = 'b', linewidth = 2)
    ax.axvline(5008.240,linestyle='--', alpha=0.3, color = 'b', linewidth = 2)
    # Balmer
    ax.axvline(4102.89 ,linestyle='--', alpha=0.3, color = 'r', linewidth = 2)
    ax.axvline(4341.68 ,linestyle='--', alpha=0.3, color = 'r', linewidth = 2)
    ax.axvline(4862.68 ,linestyle='--', alpha=0.3, color = 'r', linewidth = 2)
    ax.axvline(6564.61 ,linestyle='--', alpha=0.3, color = 'r', linewidth = 2)

    ax.axvline(6718.29,linestyle='--', alpha=0.3, color = 'b', linewidth = 2) #SII
#     ax.axvline(2799.117,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) # MgII

    ax.axvline(3934,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) # H
    ax.axvline(3969,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) # K
    ax.axvline(4305,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) # G

    ax.axvline(5176.7,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) # Mg
    ax.axvline(5895.6,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) #Na
    ax.axvline(8500,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) #CaII
    ax.axvline(8544,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) #CaII
    ax.axvline(8664,linestyle='-', alpha=0.3, color = 'k', linewidth = 2) #CaII
    ax.legend(loc=4, fontsize=15)

def stack_phot(DB):
    PWV = []
    PER = []
    PFL = []

    for ID in DB.index:
        pwv,pfl,per,pnum = np.load('../phot/{0}_{1}_phot.npy'.format(DB.field[ID], DB.id[ID]))

        pwv = pwv[pfl > 0] 
        per = per[pfl > 0] 
        pfl = pfl[pfl > 0]

        pfl = pfl[np.argsort(pwv)]
        per = per[np.argsort(pwv)]
        pwv = pwv[np.argsort(pwv)]

        IDX = [U for U in range(len(pwv)) if 4000 < pwv[U] < 18000]
        
        norm = np.trapz(pfl[IDX],pwv[IDX]/ (1 +DB.zgrism[ID] ))

        for w in range(len(pwv)):
            if 7000 < pwv[w]/ (1 +DB.zgrism[ID] ) < 9000 and pfl[w] /norm < 5E-5:
                print(DB.field[ID], DB.id[ID])
        
        PWV.extend(pwv / (1 +DB.zgrism[ID] ))
        PER.extend(per/norm)
        PFL.extend(pfl/norm)

        
    PWV = np.array(PWV)
    PFL = np.array(PFL)
    PER = np.array(PER)
    
    return PWV, PFL, PER

def Leave_one_out_STACK(wave, flxgrid, errgrid):
    
    Y,Ye = Stack(wave, flxgrid, errgrid)

    weights = np.zeros([len(flxgrid), len(wave)])
    for i in range(len(flxgrid)):
        fgrid = []
        egrid = []
        for ii in range(len(flxgrid)):
            if i != ii:
                fgrid.append(flxgrid[ii])
                egrid.append(errgrid[ii])  
        Ybar, Ybare = Stack(wave, fgrid, egrid)
        weights[i] = np.abs(Ybar - Y) / Y
    return weights
##########################################################



Adb = pd.read_pickle('../dataframes/fitdb/evolution_db_masslim.pkl')
Adb = Adb.query('id != 44707')
pdb = Adb.sort_values('sf_prob_ml', ascending = False)
pdb.reset_index(drop=True, inplace=True)


idx = 0

l_n = ['OII', 'HK','H$\delta$', 'G H$\gamma$', 'H$\\beta$', 'OIII', 'Mg', 'Na', 'H$\\alpha$ + NII', 'SII']
l_x = [3630,3850, 4050, 4250, 4700, 4900, 5150, 5800, 6200, 6750]
swave = np.arange(1000,10000,15)

while len(pdb.index[0 + idx:100 + idx]) == 100:    
    s1_flx, s1_err = stack_setup(pdb.query('{} < index < {}'.format(0 + idx,100 + idx)), swave)
    
    mn = 1
    mx = 0

    for i in range(len(s1_flx)):
        f = s1_flx[i]
        if 3400 < swave[i] < 8200:
            
            if f < mn:
                mn = f
            if f > mx:
                mx = f
    
    mx -= mn
    
    s1_flx -= mn
    s1_err /= mx
    s1_flx /= mx
    
    plt.figure(figsize=[18,6])
    ax1=plt.subplot()

    plot_spec(ax1, swave, s1_flx, s1_err, 3400, 8200, color = CMAP((353 - idx)/353),
              label = '{:1.3f} '.format(pdb.sf_prob_ml[0 + idx]) + '< P$_{sf}$ < ' + '{:1.3f}'.format(pdb.sf_prob_ml[100 + idx]))

    ax1.set_yticklabels([])
    ax1.set_xlim(3300,8250)
    ax1.set_xlabel('Wavelength ($\AA$)', fontsize = 25)
    ax1.set_ylabel('F$_\lambda$', fontsize = 25)
    ax1.legend(fontsize = 17)
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax1.set_ylim(-0.05,1.05)

    for i in range(len(l_n)):
        ax1.text(l_x[i], 1.08, l_n[i], fontsize=17)

    plt.savefig('../plots/evolution_plots/stack_gif/stack_{}.png'.format(idx), bbox_inches = 'tight')    
  
    idx +=1
    
#     if idx == 245:
#         break