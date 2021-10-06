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
from spec_tools import Rescale_SF_sfh, lbt_to_z, Posterior_SF_spec, IMG_pull

from time import time
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)


### set home for files
hpath = os.environ['HOME'] + '/'


import make_sfh_tool
from importlib import reload
reload(make_sfh_tool)

from make_sfh_tool import Gen_SFH
Edb = pd.read_pickle('../dataframes/fitdb/emission_line_galaxies_prior2_v1.pkl')

for i in Edb.index:
    gid = Edb.id[i]
    field = Edb.field[i]
    if not os.path.isfile('../plots/emission_line_gal/{}_{}.png'.format(field, gid)):
        zgrizli = Edb.zgrism[i]
        SFH = make_sfh_tool.Gen_SFH_p2(field, gid, zgrizli)

        spec = np.load('../full_specs/{}_{}_fullspec_p2.npy'.format(field, gid), allow_pickle = True).item()
        param = np.load('../data/posteriors/{}_{}_SFfit_p2_fits.npy'.format(field, gid), allow_pickle = True).item()

        IDB = [U for U in range(len(spec['wave'])) if min(spec['Pwv']/(1+zgrizli)) < spec['wave'][U] < min(spec['Bwv']/(1+zgrizli))]
        if len(spec['Rwv']) > 0:
            IDR = [U for U in range(len(spec['wave'])) if max(spec['Rwv']/(1+zgrizli)) < spec['wave'][U] < max(spec['Pwv']/(1+zgrizli))]
        else:
            IDR = [U for U in range(len(spec['wave'])) if max(spec['Bwv']/(1+zgrizli)) < spec['wave'][U] < max(spec['Pwv']/(1+zgrizli))]

            
        gs = gridspec.GridSpec(2,3, hspace=0.3)

        plt.figure(figsize=[20,10])
        ax1 = plt.subplot(gs[0,:]) ; ax2 = plt.subplot(gs[1,0]) ; ax3 = plt.subplot(gs[1,1]) ; ax4 = plt.subplot(gs[1,2])


        ax1.errorbar(spec['Bwv']/(1+zgrizli),spec['Bfl'] * 1E18,spec['Ber']* 1E18, linestyle = 'none',
                   marker='o', markersize=3, color='#377eb8',zorder = 2)
        if len(spec['Rfl']):
            ax1.errorbar(spec['Rwv']/(1+zgrizli),spec['Rfl'] * 1E18,spec['Rer']* 1E18, linestyle = 'none',
                marker='o', markersize=3, color='#e41a1c',zorder = 2)

        ax1.errorbar(spec['Pwv']/(1+zgrizli),spec['Pfl'] * 1E18,spec['Per']* 1E18,
                linestyle='None', marker='o', markersize=10, color='#4daf4a',zorder = 1)

        ax1.plot(spec['Bwv']/(1+zgrizli),spec['Bmfl'] * 1E18, 'k', linewidth = 2, zorder = 10)
        if len(spec['Rfl']):
            ax1.plot(spec['Rwv']/(1+zgrizli),spec['Rmfl'] * 1E18, 'k', linewidth = 2, zorder = 10)
        ax1.plot(spec['wave'][IDB],spec['flam'][IDB] * 1E18, 'k', linewidth = 2, zorder = 10)
        ax1.plot(spec['wave'][IDR],spec['flam'][IDR] * 1E18, 'k', linewidth = 2, zorder = 10, 
                label = '{}-{}, z={}'.format(field, gid, np.round(zgrizli, 3)))

        ax1.legend(loc = 1, fontsize = 15)
        ax1.set_xscale('log')
        ax1.set_xticks([2500,5000,7500,10000,25000])
        ax1.set_xticklabels([2500,5000,7500,10000,25000])
        ax1.minorticks_off()

        ax1.set_xlim(min(spec['Pwv']/(1+zgrizli))*0.95, max(spec['Pwv']/(1+zgrizli))*1.05)
        ymax = max(spec['Pfl'])
        if ymax < max(spec['Bfl']):
            ymax = max(spec['Bfl'])
        if len(spec['Rfl']):
            if ymax < max(spec['Rfl']):
                ymax = max(spec['Rfl'])
        ax1.set_ylim(-0.05,ymax * 1.1* 1E18)
        ######################
        iP = interp1d(param['lm'], param['Plm'])
        ax2.plot(param['lm'], param['Plm'], linewidth = 3, color = 'k')
        ax2.stem(Edb.query('id == {}'.format(gid)).lmass.values,iP(Edb.query('id == {}'.format(gid)).lmass.values),
                markerfmt = 'none')
        HDRL = Edb.query('id == {}'.format(gid)).lmass_lower.values[0]
        HDRH = Edb.query('id == {}'.format(gid)).lmass_upper.values[0]
        ax2.fill_between(np.linspace(HDRL, HDRH), iP(np.linspace(HDRL, HDRH)), alpha = 0.3)
        ######################
        iP = interp1d(param['m'], param['Pm'])
        ax3.plot(param['m'], param['Pm'], linewidth = 3, color = 'k')
        ax3.stem(Edb.query('id == {}'.format(gid)).Z.values,iP(Edb.query('id == {}'.format(gid)).Z.values),
                markerfmt = 'none')
        HDRL = Edb.query('id == {}'.format(gid)).Z_lower.values[0]
        HDRH = Edb.query('id == {}'.format(gid)).Z_upper.values[0]
        ax3.fill_between(np.linspace(HDRL, HDRH), iP(np.linspace(HDRL, HDRH)), alpha = 0.3)
        ######################
        iP = interp1d(SFH.log_sSFR, SFH.Plog_sSFR)
        ax4.plot(SFH.log_sSFR, SFH.Plog_sSFR, linewidth = 3, color = 'k')
        ax4.stem([SFH.lssfr],[iP(SFH.lssfr)],
                markerfmt = 'none')
        HDRL = SFH.lssfr_hci[0]
        HDRH = SFH.lssfr_hci[1]
        ax4.fill_between(np.linspace(HDRL, HDRH), iP(np.linspace(HDRL, HDRH)), alpha = 0.3)

        ax1.set_xlabel('Wavelength ($ \\rm\AA$)', fontsize=22)
        ax1.set_ylabel('F$_\lambda$ ($10^{-18}$ $erg/s/cm^{2}/\AA $)', fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=17)

        ax2.set_xlabel('log(M$_*$/M$_\odot$)', fontsize=20)
        ax2.set_ylabel('P(log(M$_*$/M$_\odot$))', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=15)

        ax3.set_xlabel('Metallicity (Z$_\odot$)', fontsize=20)
        ax3.set_ylabel('P(Z$_\odot$)', fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=15)

        ax4.set_xlabel('log(sSFR (yr$^{-1}$))', fontsize=20)
        ax4.set_ylabel('P(log(sSFR))', fontsize=20)
        ax4.tick_params(axis='both', which='major', labelsize=15)

        plt.savefig('../plots/emission_line_gal/{}_{}.png'.format(field, gid) ,bbox_inches = 'tight')
