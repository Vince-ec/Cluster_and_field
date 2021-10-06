import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import fsps
from matplotlib import gridspec
from astropy.io import fits
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

from time import time
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)

Adb = pd.read_pickle('../dataframes/fitdb/evolution_db_masslim.pkl')

from spec_id import Calibrate_grism, Scale_model
def Best_fit_scale(wv, fl, er, mfl, p1):
    cal = Calibrate_grism([wv, fl, er], mfl, p1)
    scale = Scale_model(fl / cal, er/ cal, mfl)
    FL =  fl/ cal/ scale
    ER =  er/ cal/ scale
    return FL, ER

def Q_spec_adjust(Gs, bestfits):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl= BFS

    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
    
    BFL, BER = Best_fit_scale(wvs[0], flxs[0], errs[0], Gmfl[0], bp1)
    RFL, RER = Best_fit_scale(wvs[1], flxs[1], errs[1], Gmfl[1], rp1)
    
    return BFL, BER, RFL, RER, Gmfl, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm

def SF_spec_adjust(Gs, bestfits, spz):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl= BFS

    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)
    print()
    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + spz), spz, 0, sp.stellar_mass)*10**lm, spz, 
                                    wvs, flxs, errs, beams, trans)
    
    BFL, BER = Best_fit_scale(wvs[0], flxs[0], errs[0], Gmfl[0], bp1)
    RFL, RER = Best_fit_scale(wvs[1], flxs[1], errs[1], Gmfl[1], rp1)
    
    return BFL, BER, RFL, RER, Gmfl, wave, F_lam_per_M(flux,wave*(1 + spz), spz, 0, sp.stellar_mass)*10**lm  
        
#############

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

for i in Adb.index:
    plt.figure(figsize = [15,10])
    gids = Adb.id[i]
    ax1 = plt.subplot()
    ax2 = inset_axes(ax1, width=2, height=1.5,loc = 'upper right')#bbox_to_anchor=(720, 615))

    fld = Adb.query('id == {}'.format(gids)).field.values[0]
    gal = Adb.query('id == {}'.format(gids)).id.values[0]
    lssfr = Adb.query('id == {}'.format(gids)).log_ssfr.values[0]

    sp = np.load('../full_specs/{}_{}_fullspec.npy'.format(fld, gal), allow_pickle=True).item()
    rshift = Adb.query('field == "{}" and id == {}'.format(fld, gal)).zgrism.values[0]

    if len(sp['Bwv']) > 0:
        IDX = [U for U in range(len(sp['Bwv'])) if 8200 < sp['Bwv'][U] < 15000]
        ax1.errorbar(sp['Bwv'][IDX]/(1+rshift), sp['Bfl'][IDX]*1E18, sp['Ber'][IDX]*1E18, fmt = 'bo' ,markersize=2)
        ax1.plot(sp['Bwv'][IDX]/(1+rshift), sp['Bmfl'][IDX]*1E18, 'k',
                 label = '{}{}-{}, z = {:1.3f}\nlog(sSFR) = {:1.2f}'.format(fld[0], fld[1], gal, rshift, lssfr))

    if len(sp['Rwv']) > 0:
        ax1.errorbar(sp['Rwv']/(1+rshift), sp['Rfl']*1E18, sp['Rer']*1E18, fmt = 'ro' ,markersize=2)
        ax1.plot(sp['Rwv']/(1+rshift), sp['Rmfl']*1E18, 'k')


    ax2.plot(sp['wave'], sp['flam']*1E18,'k', linewidth = 0.5,zorder = 0)
    ax2.errorbar(sp['Pwv']/(1+rshift), sp['Pfl']*1E18, sp['Per']*1E18, fmt = 'go', markersize = 4, zorder=1)

    ax2.set_xlim(0.95 * min(sp['Pwv']/(1+rshift)), 1.05 * max(sp['Pwv']/(1+rshift)))
    #     ax1.legend(fontsize = 15)
    ax1.set_xlabel(r'Rest-Frame Wavelength ($\AA$)',size=20)
    ax1.set_ylabel(r'F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=20)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_xscale('log')
    ax1.minorticks_off()
    ax1.set_title('{}{}-{}, z = {:1.3f}, log(sSFR) = {:1.2f}'.format(fld[0], fld[1], gal, rshift, lssfr),
                 fontsize = 18)
    plt.title('{}-{}, P(sf) = {:.2f}, z = {:.2f}'.format(Adb.field[i], Adb.id[i], Adb.sf_prob_ml[i],  Adb.zgrism[i]))
    plt.savefig('../plots/evolution_plots/spec_plots/{}-{}_v2.png'.format(Adb.field[i], Adb.id[i]), bbox_inches = 'tight')    
        