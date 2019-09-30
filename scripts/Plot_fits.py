import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
import fsps
from matplotlib.gridspec import GridSpec
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
from spec_tools import Photometry, Posterior_spec, Posterior_SF_spec
from spec_exam import Gen_spec, Gen_SF_spec
import pickle
import re
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

def plot_posterior(field, galaxy, param, ext, name, x_name, y_name, DB, roundto = 3):
    grow = DB.query('id == {0}'.format(galaxy))
    z,pz = np.load('../data/posteriors/{}_{}_{}_P{}.npy'.format(field, galaxy, ext, param))
    ipz = interp1d(np.round(z,roundto),pz)

    plt.plot(z,pz,'k')
    for i in range(len(grow['{0}_hci'.format(name)].values[0])//2):
        hdr = np.linspace(np.round(grow['{0}_hci'.format(name)].values[0][2*i],roundto),
                          np.round(grow['{0}_hci'.format(name)].values[0][2*i+1],roundto))
        plt.fill_between(hdr, ipz(hdr), color = '#4E7577', alpha=0.75)
    plt.vlines(grow['{0}'.format(name)].values[0],0, ipz(np.round(grow['{0}'.format(name)].values[0],roundto)), color = '#C1253C')
    plt.xlabel('{0}'.format(x_name), fontsize=15)
    plt.ylabel('P({0})'.format(y_name), fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.ylim(0,max(pz)*1.05)
    if max(pz) >= 3:
        plt.yticks(np.linspace(0,max(pz)*1.05,4).astype(int))

def PLOT(field, galaxy, DF, savefig = True):
    grow = DF.query('id == {0}'.format(galaxy))

    flist = glob('../data/posteriors/{}_{}_*_Pm.npy'.format(field, galaxy))
    for f in flist:
        ext = re.split('{}_{}_'.format(field, galaxy),
        re.split('_Pm.npy', os.path.basename(f))[0])[1]
        if ext  == 'tabfit':
            Gs = Gen_spec(field, galaxy, grow.zgrism.values[0], phot_errterm = 0.04, irac_err = 0.08) 
            Flam = Posterior_spec(field, galaxy)
            break
        if ext in 'SFfit_p1':
            Gs = Gen_SF_spec(field, galaxy, grow.z_grizli.values[0], phot_errterm = 0.04, irac_err = 0.08) 
            Flam = Posterior_SF_spec(field, galaxy, grow.z_grizli.values[0])
            break

    
    x,px = np.load('../data/posteriors/{}_{}_{}_Pbp1.npy'.format(field, galaxy, ext))
    bp1 = x[px == max(px)][0]
    x,px = np.load('../data/posteriors/{}_{}_{}_Prp1.npy'.format(field, galaxy, ext))
    rp1 = x[px == max(px)][0]
    
    Gs.Best_fit_scale_flam(Flam.wave, Flam.SPEC, Flam.rshift, bp1, rp1)

    with open('../data/SFH/{}_{}.pkl'.format(field, galaxy), 'rb') as sfh_file:
        sfh = pickle.load(sfh_file)

    gs = GridSpec(3,5, hspace=0.3, wspace = 0.35)   

    plt.figure(figsize=[19,15])
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
    plt.xlim(np.log10(min(Gs.Pwv_rf)*0.95),np.log10(max(Gs.Pwv_rf)*1.05))

    plt.xticks(np.log10([2500,5000,7500,10000,25000]),[2500,5000,7500,10000,25000])
    plt.title(galaxy, fontsize=25)
    plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
    plt.ylabel(r'F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/ \rm \AA $)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ###############sfh plot################
    isfhl = interp1d(sfh.LBT,sfh.SFH_16)
    isfhh = interp1d(sfh.LBT,sfh.SFH_84)

    ax1 = plt.subplot(gs[0,3:5])
    ax2 = ax1.twiny()

    ax1.plot(sfh.fulltimes, sfh.sfr_grid.T[:1000], color = '#532436', alpha=.075, linewidth = 0.5)
    ax1.plot(sfh.LBT,sfh.SFH, color = '#C1253C', linewidth = 3, zorder = 9)
    ax1.plot(sfh.LBT,sfh.SFH_16, 'k', linewidth = 2)
    ax1.plot(sfh.LBT,sfh.SFH_84, 'k', linewidth = 2)

    max_lbt = np.round(sfh.fulltimes[-1])
    max_age = max_lbt + cosmo.lookback_time(grow.zgrism.values[0]).value
    age_at_z = cosmo.lookback_time(grow.zgrism.values[0]).value
    
    if ext == 'tabfit':
        zarray = [np.round(grow.zgrism.values[0],2)]
    else:
        zarray = [np.round(grow.z_grizli.values[0],2)]
        
    idx = 0
    while cosmo.lookback_time(np.array(zarray[idx])).value  < 13:
        zarray.append(int(zarray[idx])+1)
        idx += 1 
    
    tarray =cosmo.lookback_time(np.array(zarray)).value - cosmo.lookback_time(np.array(zarray)).value[0]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tarray)
    ax2.set_xticklabels(zarray)
    ax2.xaxis.set_ticks_position('top')

    ax1.set_xlabel('Look-back time (Gyr)', fontsize=21)
    ax1.set_ylabel('SFR ($M_\odot$ / yr)', fontsize=21)
    ax2.set_xlabel('Redshift (z)', fontsize=21) 
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=17)

    ax1.vlines(grow.t_50.values[0],isfhl(grow.t_50.values[0]), isfhh(grow.t_50.values[0]), color = '#ED2D39', linewidth = 2, zorder = 11)
    ax1.vlines(grow.t_50.values[0],isfhl(grow.t_50.values[0]), isfhh(grow.t_50.values[0]), color = 'k', linewidth = 4, zorder = 10)

    hdr = np.linspace(grow.t_50_hci.values[0][0],grow.t_50_hci.values[0][1])

    ax1.fill_between(hdr, isfhh(hdr), isfhl(hdr), color = '#4E7577', alpha=0.75, zorder = 8)
    ax1.vlines(grow.t_50_hci.values[0][0],isfhl(grow.t_50_hci.values[0][0]),isfhh(grow.t_50_hci.values[0][0]), 
               color = 'k', linewidth = 1, zorder = 8)
    ax1.vlines(grow.t_50_hci.values[0][1],isfhl(grow.t_50_hci.values[0][1]),isfhh(grow.t_50_hci.values[0][1]), 
               color = 'k', linewidth = 1, zorder = 8)    
    ###############plot zoom tab##################
    plt.subplot(gs[1,:3])

    if Gs.g102:
        plt.errorbar(Gs.Bwv_rf,Gs.Bfl *1E18,Gs.Ber *1E18,
                linestyle='None', marker='o', markersize=3, color='#36787A', zorder = 2)
        plt.plot(Gs.Bwv_rf,Gs.Bmfl *1E18,'k', zorder = 4)
        
    if Gs.g141:
        plt.errorbar(Gs.Rwv_rf,Gs.Rfl *1E18,Gs.Rer *1E18,
                linestyle='None', marker='o', markersize=3, color='#EA2E3B', zorder = 2)
        plt.plot(Gs.Rwv_rf,Gs.Rmfl *1E18,'k', zorder = 4)

    plt.xlabel(r'Wavelength ($\rm  \AA$)', fontsize=20)
    plt.ylabel(r'F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/ \rm \AA $)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
        
    #################plot uvj###################    
    plt.subplot(gs[1,3:5])
    plt.scatter(Bdb.VJ, Bdb.UV)
    plt.scatter(grow.VJ.values, grow.UV.values, marker = '*', color = 'r', s = 100)
    
    if grow.AGN.values == 'AGN':
        plt.scatter(grow.VJ.values, grow.UV.values,marker = 'o', s=500,linewidth = 5, color = 'None', edgecolor = 'g')
        
    plt.plot([0,.9],[1.382,1.382],'k',lw=1.2)
    plt.plot([1.65,1.65],[2.05,2.5],'k',lw=1.2)
    plt.plot([.9,1.65],[0.88*.9+0.59,0.88*1.65+0.59],'k',lw=1.2)
    plt.xlabel('(V-J)', fontsize=21)
    plt.ylabel('(U-V)', fontsize=21)
    plt.tick_params(axis='both', which='major', labelsize=17)

    ###############P(Z)################
    plt.subplot(gs[2,0])
    plot_posterior(field, galaxy, 'm', ext, 'Z', 'Z / Z$_\odot$', 'Z / Z$_\odot$', Bdb)
    
    ###############P(lwa)################
    plt.subplot(gs[2,1])
    plot_posterior(field, galaxy, 'lwa', ext, 'lwa', 'LWA (Gyr)', 'LWA', Bdb)
 
    ###############P(z)################
    plt.subplot(gs[2,2])
    if ext == 'tabfit':
        plot_posterior(field, galaxy, 'z', ext, 'zgrism', 'Redshift (z)', 'z', Bdb, roundto=4)
    
    ###############P(d)################
    plt.subplot(gs[2,3])
    plot_posterior(field, galaxy, 'd', ext, 'Av', 'Dust (Av)', 'Av', Bdb)
    
    ###############P(lmass)################
    plt.subplot(gs[2,4])
    plot_posterior(field, galaxy, 'lm', ext, 'lmass', 'log(M/M$_\odot$)', 'log(M/M$_\odot$)', Bdb)
    
    if savefig:
        plt.savefig('../plots/bulgefits/all_data_{0}_{1}.png'.format(field, galaxy),bbox_inches = 'tight')
    
Bdb = pd.read_pickle('../dataframes/fitdb/buldgefitsdb.pkl')

for i in Bdb.index:
    if not os.path.isfile('../plots/bulgefits/all_data_{0}_{1}.png'.format(Bdb.field[i], Bdb.id[i])):
        try:
            PLOT(Bdb.field[i], Bdb.id[i], Bdb,)
        except:
            pass