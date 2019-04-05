import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
import fsps
from matplotlib.gridspec import GridSpec
from spec_tools import Source_present, Scale_model, Oldest_galaxy, Sig_int,Likelihood_contours, Median_w_Error_cont
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
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, \
    bp1, rp1, ba, bb, bl, ra, rb, rl, lwa, logl = np.load(
        '../data/bestfits/{0}_{1}_tabfit_bfit.npy'.format(field, galaxy))
    
    sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr)    
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)    
    
    Gs = Gen_spec(field, galaxy, z, g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = False,
        phot_errterm = 0.03, irac_err = 0.06, decontam = True) 

    Gs.Sim_all_premade(wave*(1+z),flux)
    
    if Gs.g102 and Gs.g141:
        bcal, rcal = Calibrate_grism(Gs, [Gs.Bmfl,Gs.Rmfl], [bp1,rp1], [Gs.Bwv,Gs.Rwv], [Gs.Bfl,Gs.Rfl], [Gs.Ber,Gs.Rer])
    
    if Gs.g102 and not Gs.g141:
        bcal = Calibrate_grism(Gs, [Gs.Bmfl], [bp1], [Gs.Bwv], [Gs.Bfl], [Gs.Ber])[0]
    
    if not Gs.g102 and Gs.g141:
        rcal = Calibrate_grism(Gs, [Gs.Rmfl], [rp1], [Gs.Rwv], [Gs.Rfl], [Gs.Rer])[0]
    
    
    gs = GridSpec(3,4, hspace=0.3, wspace = 0.3)   

    plt.figure(figsize=[16,15])
    ###############plot tab##################
    plt.subplot(gs[0,:3])

    if Gs.g102:
        plt.errorbar(np.log10(Gs.Bwv_rf),Gs.Bfl*1E18 / bcal,Gs.Ber*1E18 / bcal,
                linestyle='None', marker='o', markersize=3, color='#377eb8', zorder = 2)
        plt.plot(np.log10(Gs.Bwv_rf),Gs.Bmfl*1E18,'k', zorder = 4)

    if Gs.g141:
        plt.errorbar(np.log10(Gs.Rwv_rf),Gs.Rfl*1E18 /  rcal,Gs.Rer*1E18 / rcal,
                linestyle='None', marker='o', markersize=3, color='#e41a1c', zorder = 2)
        plt.plot(np.log10(Gs.Rwv_rf),Gs.Rmfl*1E18,'k', zorder = 4)
    
    plt.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='p', markersize=15, color='#984ea3', zorder = 1)
    plt.plot(np.log10(Gs.Pwv_rf),Gs.Pmfl*1E18,'ko', zorder = 3)
    plt.xticks(np.log10([2500,5000,7500,10000,25000]),[2500,5000,7500,10000,25000])
    plt.title(galaxy, fontsize=25)
    plt.xlabel('Wavelength ($\AA$)', fontsize=20)
    plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ###############sfh plot################
    med,le,he = np.zeros([3,10])
    for i in range(10):
        x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pm{2}.npy'.format(field, galaxy,i+1))
        med[i],le[i],he[i] = Highest_density_region(px,x)

    x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pa.npy'.format(field, galaxy))
    a,ale,ahe = Highest_density_region(px,x)

    time, sfr, tmax = convert_sfh(get_agebins(a), med, maxage = a*1E9)
    time, sfr_l, tmax = convert_sfh(get_agebins(a), med - le, maxage = a*1E9)
    time, sfr_h, tmax = convert_sfh(get_agebins(a), med + he, maxage = a*1E9) 
    
    plt.subplot(gs[0,3])
    plt.plot(time,sfr, 'r')
    plt.fill_between(time,sfr_l, sfr_h, alpha = 0.3)
    plt.xlabel('Age', fontsize=15)
    plt.ylabel('SFR', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    ###############plot delay################
    sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = 0, sfh = 4, tau = 0.01, dust_type = 1)
    m, a, tau, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa, logl = np.load(
        '../data/bestfits/{0}_{1}_delayfit_bfit.npy'.format(field, galaxy))
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    sp.params['tau'] = tau
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)    
    
    Gs = Gen_spec(field, galaxy, z, g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = False,
        phot_errterm = 0.05, irac_err = 0.1, decontam = True) 

    Gs.Sim_all_premade(wave*(1+z),flux)
    
    if Gs.g102 and Gs.g141:
        bcal, rcal = Calibrate_grism(Gs, [Gs.Bmfl,Gs.Rmfl], [bp1,rp1], [Gs.Bwv,Gs.Rwv], [Gs.Bfl,Gs.Rfl], [Gs.Ber,Gs.Rer])
    
    if Gs.g102 and not Gs.g141:
        bcal = Calibrate_grism(Gs, [Gs.Bmfl], [bp1], [Gs.Bwv], [Gs.Bfl], [Gs.Ber])[0]
    
    if not Gs.g102 and Gs.g141:
        rcal = Calibrate_grism(Gs, [Gs.Rmfl], [rp1], [Gs.Rwv], [Gs.Rfl], [Gs.Rer])[0]
        
    plt.subplot(gs[1,:3])

    if Gs.g102:
        plt.errorbar(np.log10(Gs.Bwv_rf),Gs.Bfl*1E18 / bcal,Gs.Ber*1E18 / bcal,
                linestyle='None', marker='o', markersize=3, color='#377eb8', zorder = 2)
        plt.plot(np.log10(Gs.Bwv_rf),Gs.Bmfl*1E18,'k', zorder = 4)

    if Gs.g141:
        plt.errorbar(np.log10(Gs.Rwv_rf),Gs.Rfl*1E18 /  rcal,Gs.Rer*1E18 / rcal,
                linestyle='None', marker='o', markersize=3, color='#e41a1c', zorder = 2)
        plt.plot(np.log10(Gs.Rwv_rf),Gs.Rmfl*1E18,'k', zorder = 4)
    
    plt.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='p', markersize=15, color='#984ea3', zorder = 1)
    plt.plot(np.log10(Gs.Pwv_rf),Gs.Pmfl*1E18,'ko', zorder = 3)
    plt.xticks(np.log10([2500,5000,7500,10000,25000]),[2500,5000,7500,10000,25000])
    plt.xlabel('Wavelength ($\AA$)', fontsize=20)
    plt.ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)

    ###############sfh plot################
    x,px = np.load('../data/posteriors/{0}_{1}_delayfit_Pt.npy'.format(field, galaxy))
    tau, taul, tauh = Highest_density_region(px,x)

    x,px = np.load('../data/posteriors/{0}_{1}_delayfit_Pa.npy'.format(field, galaxy))
    a,ale,ahe = Highest_density_region(px,x)

    t = np.arange(0,a,0.01)
    sfr = t*np.exp( -t / tau) / np.trapz(t*np.exp( -t / tau),t)
    sfrl = t*np.exp( -t /(tau - taul)) / np.trapz(t*np.exp( -t /(tau - taul)),t)
    sfrh = t*np.exp( -t / (tau + tauh)) /  np.trapz(t*np.exp( -t / (tau + tauh)),t)
   
    plt.subplot(gs[1,3])
    plt.plot(t,sfr,'b')
    plt.fill_between(t,sfrl, sfrh, alpha = 0.3)
    plt.xlabel('Age', fontsize=15)
    plt.ylabel('SFR', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    ###############P(Z)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pm.npy'.format(field, galaxy))
    d,pd = np.load('../data/posteriors/{0}_{1}_delayfit_Pm.npy'.format(field, galaxy))
    
    plt.subplot(gs[2,0])
    plt.plot(z,pz,'r')
    plt.plot(d,pd,'b')
    plt.xlabel('Z / Z$_\odot$', fontsize=15)
    plt.ylabel('P(Z)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    ###############P(lwa)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Plwa.npy'.format(field, galaxy))
    d,pd = np.load('../data/posteriors/{0}_{1}_delayfit_Plwa.npy'.format(field, galaxy))

    plt.subplot(gs[2,1])
    plt.plot(z,pz,'r')
    plt.plot(d,pd,'b')
    plt.xlabel('Age', fontsize=15)
    plt.ylabel('P(lwa)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    ###############P(z)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pz.npy'.format(field, galaxy))
    d,pd = np.load('../data/posteriors/{0}_{1}_delayfit_Pz.npy'.format(field, galaxy))
    
    plt.subplot(gs[2,2])
    plt.plot(z,pz,'r')
    plt.plot(d,pd,'b')
    plt.xlabel('redshift', fontsize=15)
    plt.ylabel('P(z)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    ###############P(d)################
    z,pz = np.load('../data/posteriors/{0}_{1}_tabfit_Pd.npy'.format(field, galaxy))
    d,pd = np.load('../data/posteriors/{0}_{1}_delayfit_Pd.npy'.format(field, galaxy))
    
    plt.subplot(gs[2,3])
    plt.plot(z,pz,'r')
    plt.plot(d,pd,'b')
    plt.xlabel('Av', fontsize=15)
    plt.ylabel('P(Av)', fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    
    if savefig:
        plt.savefig('../plots/fullfits/all_data_{0}_{1}.png'.format(field, galaxy),bbox_inches = 'tight')
        
    
flist = glob('../data/posteriors/*zfit*')

fld = [os.path.basename(U).split('_')[0] for U in flist] 
ids = np.array([os.path.basename(U).split('_')[1] for U in flist]).astype(int)

for i in range(len(flist)):
    PLOT(fld[i], ids[i])