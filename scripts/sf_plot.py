import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import fsps
from matplotlib.gridspec import GridSpec
from spec_exam import Gen_SF_spec, Gen_spec, Gen_spec_2D
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as sea
import os

import seaborn as sea
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)

Cdb = pd.read_pickle('../dataframes/fitdb/emission_line_galaxies.pkl')
sfdb = pd.read_pickle('../dataframes/fitdb/SFfits_1D.pkl')

from spec_id import Calibrate_grism, Scale_model
def Best_fit_scale(wv, fl, er, mfl, p1):
    cal = Calibrate_grism([wv, fl, er], mfl, p1)
    scale = Scale_model(fl / cal, er/ cal, mfl)
    FL =  fl/ cal/ scale
    ER =  er/ cal/ scale
    return FL, ER

from spec_id import Gather_grism_data_from_2d

for idx in Cdb.index:
    field = Cdb.field[idx]
    galaxy = Cdb.id[idx]
    specz = Cdb.zgrism[idx]
    
    if not os.path.isfile('../plots/sfplots/{}-{}.png'.format(field, galaxy)):
        try:
            sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
            sp.params['dust1'] = 0

            ###########gen spec##########
            Gs = Gen_spec_2D(field, galaxy, specz, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                             phot_errterm = 0.04, irac_err = 0.08, mask =True)
            ####generate grism items#####
            wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

            params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 
                      'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
            BFS = []

            for i in params:
                BFS.append(sfdb.query('field == "{}" and id == {}'.format(field, galaxy))['bf{}'.format(i)].values[0])

            from spec_id import convert_sfh, Full_forward_model, Full_calibrate_2, get_agebins, F_lam_per_M

            m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl = BFS

            sp.params['dust2'] = d
            sp.params['logzsol'] = np.log10(m)

            time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

            sp.set_tabular_sfh(time,sfr) 

            wave, flux = sp.get_spectrum(tage = a, peraa = True)

            Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm, specz, 
                                                wvs, flxs, errs, beams, trans)

            Bi = 'none'
            Ri = 'none'
            if Gs.g102:
                Bi = 0
            else:
                Ri = 0

            if Gs.g141:
                Ri = 1

            try:
                BFL, BER = Best_fit_scale(wvs[Bi], flxs[Bi], errs[Bi], Gmfl[Bi], bp1)
            except:
                pass
            try:
                RFL, RER = Best_fit_scale(wvs[Ri], flxs[Ri], errs[Ri], Gmfl[Ri], rp1)
            except:
                pass

            gs = GridSpec(1,2, width_ratios=[4,1])

            plt.figure(figsize=[18,6])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            if Gs.g102:
                ax1.errorbar(np.log10(Gs.Bwv_rf),BFL *1E18,BER *1E18,
                        linestyle='None', marker='o', markersize=1, color='#36787A', zorder = 2, elinewidth = 1)
                ax1.plot(np.log10(Gs.Bwv_rf),Gmfl[Bi] *1E18,'k', zorder = 4, alpha = 0.75)
                IDB = [U for U in range(len(wave)) if wave[U] < Gs.Bwv_rf[0]]
            else:
                IDB = [U for U in range(len(wave)) if wave[U] < Gs.Rwv_rf[0]]

            if Gs.g141:
                ax1.errorbar(np.log10(Gs.Rwv_rf),RFL *1E18,RER *1E18,
                        linestyle='None', marker='o', markersize=1, color='#EA2E3B', zorder = 2, elinewidth = 1)
                ax1.plot(np.log10(Gs.Rwv_rf),Gmfl[Ri] *1E18,'k', zorder = 4, alpha = 0.75)
                IDR = [U for U in range(len(wave)) if wave[U] > Gs.Rwv_rf[-1]]
            else:
                IDR = [U for U in range(len(wave)) if wave[U] > Gs.Bwv_rf[-1]]

            ax1.errorbar(np.log10(Gs.Pwv_rf),Gs.Pflx*1E18,Gs.Perr*1E18,
                    linestyle='None', marker='o', markersize=7, markerfacecolor='#B5677D', zorder = 1,
                         markeredgecolor = '#685877',markeredgewidth = 1)
            # wave, (F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm)
            ax1.plot(np.log10(wave)[IDB],(F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm)[IDB]*1E18,'k', alpha = 0.75, zorder=5)
            ax1.plot(np.log10(wave)[IDR],(F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm)[IDR]*1E18,'k', alpha = 0.75)
            ax1.set_xlim(np.log10(min(Gs.Pwv_rf)*0.95),np.log10(max(Gs.Pwv_rf)*1.05))
            ax1.set_ylim(-0.1,max(Gs.Pflx *1E18)*1.1)
            ax1.set_xticks(np.log10([2500,5000,10000,25000]))
            ax1.set_xticklabels(np.array([2500,5000,10000,25000]))
            ax1.set_xlabel(r'Restframe Wavelength ($\rm \AA$)', fontsize=20)
            ax1.set_ylabel(r'F$_\lambda$', fontsize=20)

            ax1.tick_params(axis='both', which='major', labelsize=15)
            ax2.tick_params(axis='both', which='major', labelsize=15)
            ax2.set_xlabel('Metallicity (Z$_\odot$)', fontsize=20)
            ax2.set_ylabel('P(Z)', fontsize=20)

            sffits = np.load('/Volumes/Vince_CLEAR/fitting_params/{}_{}_SFfit_p1_fits.npy'.format(field, galaxy), allow_pickle = True).item()

            ax2.plot(sffits['m'],sffits['Pm'], linewidth = 3)

            ix = np.linspace(Cdb.Z_lower[idx],Cdb.Z_upper[idx])
            iP  = interp1d(sffits['m'],sffits['Pm'])

            plt.fill_between(ix, iP(ix), alpha = 0.2)
            plt.plot([Cdb.Z[idx],Cdb.Z[idx]],[0, iP(Cdb.Z[idx])], color = 'r')
            plt.ylim(0, max(sffits['Pm'])*1.05)
            ax1.set_title('{}-{}, z={}'.format(field, galaxy, np.round(Cdb.zgrism[idx],3)), fontsize = 20)

            plt.savefig('../plots/sfplots/{}-{}.png'.format(field, galaxy))

        except:
            pass
    else:
        print('skip')