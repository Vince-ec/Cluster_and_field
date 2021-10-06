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
  
adb = pd.read_pickle('../dataframes/fitdb/evolution_db.pkl')
Adb = adb.query('AGN != "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')
    
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

    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d,\
        bp1, rp1, ba, bb, bl, ra, rb, rl, lwa, logz= BFS

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

def SF_spec_adjust(Gs, bestfits, z):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

    m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl= BFS

    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
    
    BFL, BER = Best_fit_scale(wvs[0], flxs[0], errs[0], Gmfl[0], bp1)
    RFL, RER = Best_fit_scale(wvs[1], flxs[1], errs[1], Gmfl[1], rp1)
    
    return BFL, BER, RFL, RER, Gmfl, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm

from spec_id import convert_sfh, Full_forward_model, Full_calibrate_2, get_agebins, F_lam_per_M


gs = gridspec.GridSpec(2,2, hspace = 0.3, wspace=0.15)

plt.figure(figsize=[15,8])

LDX = 0 
TDX = 0

# gid = [36795,45994,21156,39170]
# field = ['GND','GSD','GND','GSD']
gid = [16499,45994,16499,45994]
field = ['GND','GSD','GND','GSD']
# usedb = [True, True, False,False]
# sf = ['SF', 'SF', 'Q', 'Q']
usedb = [True, True, True, True]
sf = ['SF', 'SF', 'SF', 'SF']
for i in range(4):   
    x=i
    Gs = Gen_spec_2D(field[x], gid[x],Adb.query('id == {}'.format(gid[x])).zgrism.values[0], g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                     phot_errterm = 0.04, irac_err = 0.08,)

    if usedb[x]:
        params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 
                  'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
        BFS = []

        for i in params:
            BFS.append(Adb.query('field == "{}" and id == {}'.format(field[x], gid[x]))['bf{}'.format(i)].values[0])
    else:
        BFS = np.load('../data/bestfits/{}_{}_tabfit_bfit.npy'.format(field[x], gid[x]))


    if sf[x] == 'SF':
        BFL, BER, RFL, RER, Gmfl, wave, flam = SF_spec_adjust(Gs, BFS, Gs.specz)
    else:
        BFL, BER, RFL, RER, Gmfl, wave, flam = Q_spec_adjust(Gs, BFS)

    ax = plt.subplot(gs[TDX,LDX])
    if i == 0:
        ax.errorbar(np.log10(Gs.Bwv_rf),BFL*1E18, BER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#1f8ba3', zorder = 2, elinewidth = 0.4, alpha = 1, label = 'G102')
    else:
        ax.errorbar(np.log10(Gs.Bwv_rf),BFL*1E18, BER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#1f8ba3', zorder = 2, elinewidth = 0.4, alpha = 1)
    ax.plot(np.log10(Gs.Bwv_rf),Gmfl[0] *1E18,'k', zorder = 4, alpha = 0.75)
    IDB = [U for U in range(len(wave)) if wave[U] < Gs.Bwv_rf[0]]

    if i ==0:
        ax.errorbar(np.log10(Gs.Rwv_rf),RFL*1E18, RER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#dc1f22', zorder = 2, elinewidth = 0.4, alpha = 1, label = 'G102')
    else:
        ax.errorbar(np.log10(Gs.Rwv_rf),RFL*1E18, RER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#dc1f22', zorder = 2, elinewidth = 0.4, alpha = 1)
    ax.plot(np.log10(Gs.Rwv_rf),Gmfl[1] *1E18,'k', zorder = 4, alpha = 0.75)
    IDR = [U for U in range(len(wave)) if wave[U] > Gs.Rwv_rf[-1]]

    if i==0:
        ax.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='o', markersize=10, markerfacecolor='#8a1e72', zorder = 1,
                 markeredgecolor = '#685877',markeredgewidth = 1, label = 'Photometry')
    else:
        ax.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='o', markersize=10, markerfacecolor='#8a1e72', zorder = 1,
                 markeredgecolor = '#685877',markeredgewidth = 1)

    ax.plot(np.log10(wave)[IDB],flam[IDB]*1E18,'k', alpha = 0.75, zorder=5)
    ax.plot(np.log10(wave)[IDR],flam[IDR]*1E18,'k', alpha = 0.75)
    ax.set_xlim(np.log10(min(Gs.Pwv_rf)*0.95),np.log10(max(Gs.Pwv_rf)*1.05))
    
    fmax = max(Gs.Rfl *1E18)
    
    if fmax < max(Gs.Bfl *1E18):
        fmax = max(Gs.Bfl *1E18)
        
    if fmax < max(Gs.Pflx *1E18):
        fmax = max(Gs.Pflx *1E18)
    
    
    ax.legend(title ='{}-{}, z={}'.format(field[x], gid[x], np.round(Gs.specz,3)), fontsize = 12)
    ax.set_ylim(-0.1,fmax*1.1)
    ax.set_xticks(np.log10([2500,5000,10000,25000]))
    ax.set_xticklabels(np.array([2500,5000,10000,25000]))
    ax.set_xlabel(r'Wavelength ($\rm \AA$)', fontsize=18)
    ax.set_ylabel(r'F$_\lambda$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.get_legend().get_title().set_fontsize('15')

    if LDX == 0:
        LDX += 1
    else:
        LDX = 0
        TDX +=1

plt.savefig('../plots/evolution_plots/spec_plot.png', bbox_inches = 'tight')    




