import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from matplotlib import gridspec
import matplotlib.pyplot as plt

### load in catalog to pull in values, NEED TO CHANGE TO YOUR PATH
adb = pd.read_pickle('../dataframes/fitdb/evolution_db_v2.pkl')

### set up grid for figure and figure size
gs = gridspec.GridSpec(4,1, hspace = 0.5)
plt.figure(figsize = [12,14])

### IDs of the galaxies used
gids = [39170,12078]

for i in range(len(gids)):
    ax1 = plt.subplot(gs[i*2])
    
    ### pull in values for each galaxy
    fld = adb.query('id == {}'.format(gids[i])).field.values[0]       #Field
    gal = adb.query('id == {}'.format(gids[i])).id.values[0]          #ID
    lssfr = adb.query('id == {}'.format(gids[i])).log_ssfr.values[0]  #log(sSFR)
    mass = adb.query('id == {}'.format(gids[i])).lmass.values[0]      #log(mass)
    sfr = 10**lssfr * 10**mass                                        #SFR 
    cmass = mass - 0.26                                               #mass with correcting factor
    lssfr = np.log10(sfr / 10**cmass)                                 #adjusted log(sSFR)
    rshift = adb.query('field == "{}" and id == {}'.format(fld, gal)).zgrism.values[0]     #redshift

    
    #####load in fullspectrum, NEED TO CHANGE TO YOUR PATH
    sp = np.load('../full_specs/{}_{}_fullspec.npy'.format(fld, gal), allow_pickle=True).item() 
    
    ### plot grism spectra
    if len(sp['Bwv']) > 0:
        IDX = [U for U in range(len(sp['Bwv'])) if 8200 < sp['Bwv'][U] < 15000]
        ax1.errorbar(sp['Bwv'][IDX]/(1+rshift), sp['Bfl'][IDX]*1E18, sp['Ber'][IDX]*1E18, fmt = 'bo' ,markersize=2)
        ax1.plot(sp['Bwv'][IDX]/(1+rshift), sp['Bmfl'][IDX]*1E18, 'k')

    if len(sp['Rwv']) > 0:
        ax1.errorbar(sp['Rwv']/(1+rshift), sp['Rfl']*1E18, sp['Rer']*1E18, fmt = 'ro' ,markersize=2)
        ax1.plot(sp['Rwv']/(1+rshift), sp['Rmfl']*1E18, 'k')

    ### plot photometry
    ax1.errorbar(sp['Pwv']/(1+rshift), sp['Pfl']*1E18, sp['Per']*1E18, fmt = 'go', markersize = 7)
    
    ### plot model spec
    IDB = [U for U in range(len(sp['wave'])) if sp['wave'][U] < (sp['Bwv'][IDX]/(1+rshift))[0]]
    ax1.plot(sp['wave'][IDB], sp['flam'][IDB]*1E18,
            'k', linewidth = 1,zorder = 10,
             label = '{}{}-{}, z = {:1.3f}\nlog(M/M$\odot$) = {:1.2f}'.format(fld[0], fld[1], gal, rshift, cmass))
    
    IDR = [U for U in range(len(sp['wave'])) if sp['wave'][U] > (sp['Rwv']/(1+rshift))[-1]]
    ax1.plot(sp['wave'][IDR], sp['flam'][IDR]*1E18,'k', linewidth = 1,zorder = 10)
    
    IDG = [U for U in range(len(sp['wave'])) if (sp['Bwv'][IDX]/(1+rshift))[-1] < sp['wave'][U] < (sp['Rwv']/(1+rshift))[0]]
    ax1.plot(sp['wave'][IDG], sp['flam'][IDG]*1E18,'k', linewidth = 1,zorder = 10)
    
    ### plot adjustments
    ax1.set_xlim(0.95 * min(sp['Pwv']/(1+rshift)), 1.05 * max(sp['Pwv']/(1+rshift)))
    ax1.legend(fontsize = 12)
    ax1.set_xlabel(r'Rest-Frame Wavelength ($\AA$)',size=15)
    ax1.set_ylabel(r'F$_\lambda$ (10$^{-18}$ erg/s/cm$^2$/$\AA$)',size=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xscale('log')
    ax1.set_xticks([2500, 5000, 10000, 25000])
    ax1.set_xticklabels([2500, 5000, 10000, 25000])
    ax1.minorticks_off()
    
    ax2 = plt.subplot(gs[i*2 +1])
    
    ### load in SFH, NEED TO CHANGE TO YOUR PATH
    lbt, sfh = np.load('../data/B_sfh/{}_{}.npy'.format(fld, gal))
    lbt, sfh16 = np.load('../data/B_sfh/{}_{}_16.npy'.format(fld, gal))
    lbt, sfh84 = np.load('../data/B_sfh/{}_{}_84.npy'.format(fld, gal))
        
    ### plot SFH w/ errors
    ax2.plot(lbt, sfh, 'k', linewidth = 2, label = 'SFH')
    ax2.fill_between(lbt, sfh16, sfh84, alpha = 0.2, color = 'k', label = 'Inner 68th percentile')
    
    ### plot adjustments
    ax2.legend(fontsize = 12)
    ax2.set_xlabel('Lookback time (Gyr)',size=15)
    ax2.set_ylabel('SFR (M$_\odot$ yr$^1$)',size=15)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    ### setup to add redshift on top of SFH plots
    ax3 = ax2.twiny()
      
    ### get redshift locations for ticks
    zarray = [np.round(rshift,2)]
    
    idx = 0
    while cosmo.lookback_time(np.array(zarray[idx])).value  < 12.7:
        zarray.append(int(zarray[idx])+1)
        idx += 1 

    tarray =cosmo.lookback_time(np.array(zarray)).value - cosmo.lookback_time(np.array(zarray)).value[0]

    ### plot adjustments
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xticks(tarray)
    ax3.set_xticklabels(zarray)
    ax3.xaxis.set_ticks_position('top')
    ax3.set_xlabel('Redshift (z)', fontsize=15) 
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.minorticks_off()

    
plt.savefig('spec_examples.pdf', bbox_inches = 'tight')        