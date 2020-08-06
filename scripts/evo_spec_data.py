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

alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
sfdb = pd.read_pickle('../dataframes/fitdb/SFfits_1D.pkl')

inout = []
for i in alldb.index:
    IO = 'i'
    if alldb.field[i] == 'GND' and alldb.id[i] in sfdb.query('field == "GND"').id.values:
        IO = 'o' 
    if alldb.field[i] == 'GSD' and alldb.id[i] in sfdb.query('field == "GSD"').id.values:
        IO = 'o' 
    inout.append(IO)
    
alldb['inout'] = inout

Qdb = alldb.query('inout == "i" and t_50 > 0')

adb = pd.concat([Qdb, sfdb])
adb = adb.reset_index()
adb = adb.drop(columns='index')
Adb = adb.query('AGN != "AGN" and lmass > 10 and 0.7 < zgrism < 2.7')

from scipy.optimize import curve_fit

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

y,x=np.histogram(Adb.log_ssfr.values,100)

x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

expected=(-12, 0.45, 12,-9.56, 0.649,25)
params,cov=curve_fit(bimodal,x,y,expected)
sigma=np.sqrt(np.diag(cov))
print(params)
X = np.linspace(-14, -7, 10000)
Qdist = gauss(X, -11.50173391, 0.68224873, 7.21979314)
SFdist = gauss(X, -9.46921134, 0.49696219, 20.36197415)

Qint = []
Sint = []
for i in range(len(X)):
    Qint.append(np.trapz(Qdist[i:i+2],X[i:i+2]))
    Sint.append(np.trapz(SFdist[i:i+2],X[i:i+2]))    

Fint = np.add(Qint, Sint)

SFprob = interp1d(X,Sint / Fint)

t_q = []
lSig = []
lRe = []
lZ = []
z50 = []
concen = []
SFR = []
sfprob = []
for i in adb.index:
    if adb.zgrism[i] < 0:
        adb.zgrism[i] = adb.zfit[i]

    if adb.sSFR_avg[i] > 0:
        adb.log_ssfr[i] = adb.sSFR_avg[i]
     
    t_q.append(adb.t_50[i] - adb.t_90[i])
    lSig.append(np.log10(adb.Sigma1[i]))
    lRe.append(np.log10(adb.Re[i]))
    lZ.append(np.log10(adb.Z[i]))
    z50.append(cosmo.lookback_time(adb.z_50[i]).value)
    concen.append(np.log10(adb.Sigma1[i] / (10**adb.lmass[i])))
    SFR.append(np.log10(10**adb.log_ssfr[i] * 10**adb.lmass[i]))
    sfprob.append(float(SFprob(adb.log_ssfr[i])))
    
adb['t_q'] = t_q
adb['log_Sigma1'] = lSig
adb['log_Re'] = lRe
adb['log_Z'] = lZ
adb['z50'] = z50
adb['concen'] = concen
adb['log_sfr'] = SFR
adb['sf_prob'] = sfprob
  
Adb = adb.query('AGN != "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')
Axdb = adb.query('AGN == "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')

qdb = Adb.query('log_ssfr < -11 and AGN != "AGN" and n_f < 3 and  -2.4 < concen < -0.5')
sfdb = Adb.query('log_ssfr > -11 and AGN != "AGN" and n_f < 3 and  -2.4 < concen < -0.5')
xqdb = adb.query('log_ssfr < -11 and AGN == "AGN" and n_f < 3 and  -2.4 < concen < -0.5')
xsdb = adb.query('log_ssfr > -11 and AGN == "AGN" and n_f < 3 and -2.4 < concen < -0.5')
    
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


gs = gridspec.GridSpec(2,2, hspace = 0.35, wspace=0.15)

plt.figure(figsize=[18,10])

LDX = 0 
TDX = 0

gid = [36795,45994,21156,39170]
field = ['GND','GSD','GND','GSD']
usedb = [True, True, False,False]
sf = ['SF', 'SF', 'Q', 'Q']
for i in range(4):   
    x=i
    Gs = Gen_spec_2D(field[x], gid[x],Adb.query('id == {}'.format(gid[x])).zgrism.values[0], g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                     phot_errterm = 0.04, irac_err = 0.08,)

    if usedb[x]:
        params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 
                  'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
        BFS = []

        for i in params:
            BFS.append(sfdb.query('field == "{}" and id == {}'.format(field[x], gid[x]))['bf{}'.format(i)].values[0])
    else:
        BFS = np.load('../data/bestfits/{}_{}_tabfit_bfit.npy'.format(field[x], gid[x]))


    if sf[x] == 'SF':
        BFL, BER, RFL, RER, Gmfl, wave, flam = SF_spec_adjust(Gs, BFS, Gs.specz)
    else:
        BFL, BER, RFL, RER, Gmfl, wave, flam = Q_spec_adjust(Gs, BFS)

    ax = plt.subplot(gs[TDX,LDX])

    ax.errorbar(np.log10(Gs.Bwv_rf),BFL*1E18, BER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#36787A', zorder = 2, elinewidth = 0.4)
    ax.plot(np.log10(Gs.Bwv_rf),Gmfl[0] *1E18,'k', zorder = 4, alpha = 0.75)
    IDB = [U for U in range(len(wave)) if wave[U] < Gs.Bwv_rf[0]]

    ax.errorbar(np.log10(Gs.Rwv_rf),RFL*1E18, RER*1E18,
            linestyle='None', marker='o', markersize=0.25, color='#EA2E3B', zorder = 2, elinewidth = 0.4)
    ax.plot(np.log10(Gs.Rwv_rf),Gmfl[1] *1E18,'k', zorder = 4, alpha = 0.75)
    IDR = [U for U in range(len(wave)) if wave[U] > Gs.Rwv_rf[-1]]


    ax.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
            linestyle='None', marker='o', markersize=7, markerfacecolor='#B5677D', zorder = 1,
                 markeredgecolor = '#685877',markeredgewidth = 1)

    ax.plot(np.log10(wave)[IDB],flam[IDB]*1E18,'k', alpha = 0.75, zorder=5)
    ax.plot(np.log10(wave)[IDR],flam[IDR]*1E18,'k', alpha = 0.75)
    ax.set_xlim(np.log10(min(Gs.Pwv_rf)*0.95),np.log10(max(Gs.Pwv_rf)*1.05))
    
    fmax = max(Gs.Rfl *1E18)
    
    if fmax < max(Gs.Bfl *1E18):
        fmax = max(Gs.Bfl *1E18)
        
    if fmax < max(Gs.Pflx *1E18):
        fmax = max(Gs.Pflx *1E18)
    
    
    ax.set_ylim(-0.1,fmax*1.1)
    ax.set_xticks(np.log10([2500,5000,10000,25000]))
    ax.set_xticklabels(np.array([2500,5000,10000,25000]))
    ax.set_xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
    ax.set_ylabel(r'F$_\lambda$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    ax.set_title('{}-{}, z={}'.format(field[x], gid[x], np.round(Gs.specz,3)), fontsize = 20)

    if LDX == 0:
        LDX += 1
    else:
        LDX = 0
        TDX +=1

plt.savefig('../plots/evolution_plots/spec_plot.pdf', bbox_inches = 'tight')    




