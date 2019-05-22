import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
import fsps
from matplotlib.gridspec import GridSpec
from spec_exam import Gen_spec
from spec_id import *
from sim_engine import *
from spec_stats import Highest_density_region, Linear_fit
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as sea
import os
from grizli import multifit
from grizli import model
from prospect.models.transforms import logsfr_ratios_to_masses
from time import time
import dynesty
from dynesty import plotting as dyplot
from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde
from spec_tools import Rescale_sfh, lbt_to_z, Posterior_spec

from time import time
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

def PLOT(field, galaxy, savefig = True):
    grow = morph_db.query('id == {0}'.format(galaxy))

    Gs = Gen_spec(field, galaxy, grow.zgrism.values[0], phot_errterm = 0.04, irac_err = 0.08) 
    Flam = Posterior_spec(field, galaxy)
    
    x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pbp1.npy'.format(field, galaxy))
    bp1 = x[px == max(px)][0]
    x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Prp1.npy'.format(field, galaxy))
    rp1 = x[px == max(px)][0]
    
    Gs.Best_fit_scale_flam(Flam.wave, Flam.SPEC, Flam.rshift, bp1, rp1)

    sfh = Rescale_sfh(field, galaxy)

    gs = GridSpec(2,4, hspace=0.3, wspace = 0.3)   

    plt.figure(figsize=[15,10])
    ###############plot tab##################
    plt.subplot(gs[0,:3])

    if Gs.g102:
        plt.errorbar(np.log10(Gs.Bwv_rf),Gs.Bfl *1E18,Gs.Ber *1E18,
                linestyle='None', marker='o', markersize=3, color='#36787A', zorder = 2)
        plt.plot(np.log10(Gs.Bwv_rf),Gs.Bmfl *1E18,'k', zorder = 4)
        IDB = [U for U in range(len(Flam.wave)) if Flam.wave[U] < Gs.Bwv_rf[0]]
    else:
        IDB = [U for U in range(len(Flam.wave)) if Flam.wave[U] < Gs.Rwv_rf[0]]
        
    if Gs.g141:
        plt.errorbar(np.log10(Gs.Rwv_rf),Gs.Rfl *1E18,Gs.Rer *1E18,
                linestyle='None', marker='o', markersize=3, color='#EA2E3B', zorder = 2)
        plt.plot(np.log10(Gs.Rwv_rf),Gs.Rmfl *1E18,'k', zorder = 4)
        IDR = [U for U in range(len(Flam.wave)) if Flam.wave[U] > Gs.Rwv_rf[-1]]
    else:
        IDR = [U for U in range(len(Flam.wave)) if Flam.wave[U] > Gs.Bwv_rf[-1]]

    plt.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='o', markersize=10, markerfacecolor='#B5677D', zorder = 1,
                 markeredgecolor = '#685877',markeredgewidth = 1)

    plt.plot(np.log10(Flam.wave)[IDB],Flam.SPEC[IDB]*1E18,'k', alpha = 1, label = 'Model', zorder=5)
    plt.plot(np.log10(Flam.wave)[IDR],Flam.SPEC[IDR]*1E18,'k', alpha = 1)
    plt.xlim(np.log10(1500),np.log10(50000))

    plt.xticks(np.log10([2500,5000,7500,10000,25000]),[2500,5000,7500,10000,25000])
    plt.title(galaxy, fontsize=25)
    plt.xlabel('Wavelength ($\AA$)', fontsize=20)
    plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ###############sfh plot################
    isfhl = interp1d(sfh.LBT,sfh.SFH_16)
    isfhh = interp1d(sfh.LBT,sfh.SFH_84)
    hdr = np.linspace(grow.t_50_16.values[0], grow.t_50_84.values[0])
    hdrq = np.linspace(grow.t_q_16.values[0], grow.t_q_84.values[0])

    ax1 = plt.subplot(gs[0,3])
    ax2 = ax1.twiny()

    ax1.plot(sfh.fulltimes, sfh.sfr_grid.T, color = '#532436', alpha=.075, linewidth = 0.5)
    ax1.plot(sfh.LBT,sfh.SFH, color = '#C1253C', linewidth = 2, zorder = 10)
    ax1.plot(sfh.LBT,sfh.SFH_16, 'k', linewidth = 2)
    ax1.plot(sfh.LBT,sfh.SFH_84, 'k', linewidth = 2)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.arange(0,int(sfh.fulltimes[-1])))
    ax2.set_xticklabels(np.round(lbt_to_z(np.arange(0,int(sfh.fulltimes[-1])) + cosmo.lookback_time(grow.zgrism.values[0]).value),2))
    ax2.xaxis.set_ticks_position('top')

    ax1.set_xlabel('Look-back time (Gyr)', fontsize=15)
    ax1.set_ylabel('SFR ($M_\odot$ / yr)', fontsize=15)
    ax2.set_xlabel('Redshift (z)', fontsize=15) 
    ax1.tick_params(axis='both', which='major', labelsize=10)

    ax1.fill_between(hdr, isfhh(hdr), isfhl(hdr), color = '#4E7577', alpha=0.75, zorder = 9)
    ax1.vlines(grow.t_50.values[0],isfhl(grow.t_50.values[0]), isfhh(grow.t_50.values[0]), color = '#4E7577', linewidth = 2, zorder = 12)
    ax1.vlines(grow.t_50.values[0],isfhl(grow.t_50.values[0]), isfhh(grow.t_50.values[0]), color = 'k', linewidth = 3, zorder = 11)
    ax1.vlines(grow.t_50_16.values[0],isfhl(grow.t_50_16.values[0]),isfhh(grow.t_50_16.values[0]), color = 'k', linewidth = 0.5, zorder = 9)
    ax1.vlines(grow.t_50_84.values[0],isfhl(grow.t_50_84.values[0]),isfhh(grow.t_50_84.values[0]), color = 'k', linewidth = 0.5, zorder = 9)

    ax1.fill_between(hdrq, isfhh(hdrq), isfhl(hdrq), color = '#4E7577', alpha=0.75, zorder = 9)
    ax1.vlines(grow.t_q.values[0],isfhl(grow.t_q.values[0]),isfhh(grow.t_q.values[0]), color = '#4E7577', linewidth = 2, zorder = 12)
    ax1.vlines(grow.t_q.values[0],isfhl(grow.t_q.values[0]),isfhh(grow.t_q.values[0]), color = 'k', linewidth = 3, zorder = 11)
    ax1.vlines(grow.t_q_16.values[0],isfhl(grow.t_q_16.values[0]),isfhh(grow.t_q_16.values[0]), color = 'k', linewidth = 0.5, zorder = 9)
    ax1.vlines(grow.t_q_84.values[0],isfhl(grow.t_q_84.values[0]),isfhh(grow.t_q_84.values[0]), color = 'k', linewidth = 0.5, zorder = 9)

    ###############P(Z)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pm.npy'.format(field, galaxy))

    ipz = interp1d(np.round(z,5),pz)
    hdr = np.linspace(grow.Z_16.values[0], grow.Z_84.values[0])

    plt.subplot(gs[1,0])
    plt.plot(z,pz,'k')
    plt.fill_between(hdr, ipz(hdr), color = '#4E7577', alpha=0.75)
    plt.vlines(grow.Z.values[0],0, ipz(grow.Z.values[0]), color = '#C1253C')
    plt.xlabel('Z / Z$_\odot$', fontsize=15)
    plt.ylabel('P(Z)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    ###############P(lwa)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Plwa.npy'.format(field, galaxy))

    ipz = interp1d(np.round(z,5),pz)
    hdr = np.linspace(grow.lwa_16.values[0], grow.lwa_84.values[0])

    plt.subplot(gs[1,1])
    plt.plot(z,pz,'k')
    plt.fill_between(hdr, ipz(hdr), color = '#4E7577', alpha=0.75)
    plt.vlines(grow.lwa.values[0],0, ipz(grow.lwa.values[0]), color = '#C1253C')
    plt.xlabel('Light-Weighted Age', fontsize=15)
    plt.ylabel('P(lwa)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    ###############P(z)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pz.npy'.format(field, galaxy))
    ipz = interp1d(np.round(z,5),pz)
    hdr = np.linspace(grow.zgrism_16.values[0], grow.zgrism_84.values[0])

    plt.subplot(gs[1,2])
    plt.plot(z,pz,'k')
    plt.fill_between(hdr, ipz(hdr), color = '#4E7577', alpha=0.75)
    plt.vlines(grow.zgrism.values[0],0, ipz(grow.zgrism.values[0]), color = '#C1253C')
    plt.xlabel('redshift', fontsize=15)
    plt.ylabel('P(z)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    ###############P(d)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pd.npy'.format(field, galaxy))
    ipz = interp1d(np.round(z,5),pz)
    hdr = np.linspace(grow.Av_16.values[0], grow.Av_84.values[0])
    plt.subplot(gs[1,3])
    plt.plot(z,pz,'k')
    plt.fill_between(hdr, ipz(hdr), color = '#4E7577', alpha=0.75)
    plt.vlines(grow.Av.values[0],0, ipz(grow.Av.values[0]), color = '#C1253C')
    plt.xlabel('Av', fontsize=15)
    plt.ylabel('P(Av)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)

    if savefig:
        plt.savefig('../plots/fullfits/tab_fit_{0}_{1}.png'.format(field, galaxy),bbox_inches = 'tight')

morph_db = pd.read_pickle('../dataframes/fitdb/tabfitdb.pkl')
    
for i in morph_db.index:
    PLOT(morph_db.field[i], morph_db.id[i])