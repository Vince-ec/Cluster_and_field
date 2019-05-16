import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_sfh
from glob import glob
import os

### set home for files
hpath = os.environ['HOME'] + '/'

if hpath.split('/')[-2][-1] == 'a':
    goodss_rad = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_rad = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
else:
    goodss_rad = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_rad = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()

###### create tabit db#########

select = pd.read_pickle('../spec_files/all_section.pkl')

select = select.query('AGN != "AGN" and use == True')

#####parameters to get Z, m1-10, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {}
params = ['m', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm', 'z', 'd', 'lwa']
k = ['Z', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lmass', 'zgrism', 'Av', 'lwa']

for p in range(len(params)):
    m = np.repeat(-99.0,len(select))
    l = np.repeat(0.0,len(select))
    h = np.repeat(0.0,len(select))
    
    for i in range(len(select.index)):
        try:
            x,px = np.load('../data/posteriors/{0}_{1}_tabfit_P{2}.npy'.format(select.field[select.index[i]], select.id[select.index[i]], params[p]))
            m[i], l[i], h[i] = Highest_density_region(px,x)

        except:
            pass
    
    fitvals['{0}'.format(k[p])] = np.round(m,5)
    fitvals['{0}_16'.format(k[p])] = np.round(m-l,5)
    fitvals['{0}_84'.format(k[p])] = np.round(m+h,5)
    
#make db
tabfits = pd.DataFrame(fitvals)
tabfits['field'] = select.field.values
tabfits['id'] = select.id.values

#add SFH values
z_50= np.repeat(-99.0,len(tabfits)); z_50_16= np.repeat(-99.0,len(tabfits)); z_50_84 = np.repeat(-99.0,len(tabfits))
z_q= np.repeat(-99.0,len(tabfits)); z_q_16= np.repeat(-99.0,len(tabfits)); z_q_84 = np.repeat(-99.0,len(tabfits))
t_50= np.repeat(-99.0,len(tabfits)); t_50_16= np.repeat(-99.0,len(tabfits)); t_50_84 = np.repeat(-99.0,len(tabfits))
t_q= np.repeat(-99.0,len(tabfits)); t_q_16= np.repeat(-99.0,len(tabfits)); t_q_84 = np.repeat(-99.0,len(tabfits))
log_ssfr= np.repeat(-99.0,len(tabfits)); log_ssfr_16= np.repeat(-99.0,len(tabfits)); log_ssfr_84 = np.repeat(-99.0,len(tabfits))

for i in range(len(tabfits.index)):
    try:
        sfh = Rescale_sfh(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]])

        z_50[i] = sfh.z_50
        z_50_16[i] = sfh.z_50_16
        z_50_84[i] = sfh.z_50_84
        
        z_q[i] = sfh.z_q
        z_q_16[i] = sfh.z_q_16
        z_q_84[i] = sfh.z_q_84
        
        t_50[i] = sfh.t_50
        t_50_16[i] = sfh.t_50_16
        t_50_84[i] = sfh.t_50_84
        
        t_q[i] = sfh.t_q
        t_q_16[i] = sfh.t_q_16
        t_q_84[i] = sfh.t_q_84
        
        log_ssfr[i] = sfh.lssfr
        log_ssfr_16[i] = sfh.lssfr_16
        log_ssfr_84[i] = sfh.lssfr_84
        
    except:
        pass

tabfits['z_50'] = z_50
tabfits['z_50_16'] = z_50_16
tabfits['z_50_84'] = z_50_84

tabfits['z_q'] = z_q
tabfits['z_q_16'] = z_q_16
tabfits['z_q_84'] = z_q_84

tabfits['t_50'] = t_50
tabfits['t_50_16'] = t_50_16
tabfits['t_50_84'] = t_50_84

tabfits['t_q'] = t_q
tabfits['t_q_16'] = t_q_16
tabfits['t_q_84'] = t_q_84

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_16'] = log_ssfr_16
tabfits['log_ssfr_84'] = log_ssfr_84
    
#add Reff values
Reff = []

for i in tabfits.index:

    if tabfits.field[i][1] == 'S':
        r = goodss_rad.re[goodss_rad.NUMBER == tabfits.id[i]].values * np.sqrt(goodss_rad.q[goodss_rad.NUMBER == tabfits.id[i]].values)
        Reff.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)
    if tabfits.field[i][1] == 'N':
        r = goodsn_rad.re[goodsn_rad.NUMBER == tabfits.id[i]].values * np.sqrt(goodsn_rad.q[goodsn_rad.NUMBER == tabfits.id[i]].values)
        Reff.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)

tabfits['Reff'] = np.array(Reff)

#add compactness

gamma = [0.46, 0.3, 0.21, 0.04]
beta = [0.59, 0.62, 0.63, 0.69]

masses = np.linspace(9,11.5)
compactness =[]

for i in tabfits.index:
    if 0.4 < tabfits.zgrism[i] < 1.0:
        idx = 0
    if 1.0 < tabfits.zgrism[i] < 1.5:
        idx = 1
    if 1.5 < tabfits.zgrism[i] < 2.0:
        idx = 2
    if 2.0 < tabfits.zgrism[i] < 2.5:
        idx = 3
    if np.log10(tabfits.Reff[i]) >  gamma[idx] + beta[idx]*(tabfits.lmass[i] - 11):
        compactness.append('e')
    else:
        compactness.append('c')
tabfits['compact'] = compactness

tabfits.to_pickle('../dataframes/fitdb/tabfitdb.pkl')