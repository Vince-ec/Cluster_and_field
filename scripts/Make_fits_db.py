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
    hci = []
    
    for i in range(len(select.index)):
        try:
            x,px = np.load('../data/posteriors/{0}_{1}_tabfit_P{2}.npy'.format(select.field[select.index[i]], select.id[select.index[i]], params[p]))
            m[i], dummy = Highest_density_region(px,x)
            hci.append(dummy)

        except:
            hci.append([0])
    
    fitvals['{0}'.format(k[p])] = np.round(m,5)
    fitvals['{0}_hci'.format(k[p])] = hci
    
#make db
tabfits = pd.DataFrame(fitvals)
tabfits['field'] = select.field.values
tabfits['id'] = select.id.values

#add SFH values
z_50= np.repeat(-99.0,len(tabfits))
z_50_hci = []
z_q= np.repeat(-99.0,len(tabfits)) 
z_q_hci = []
t_50= np.repeat(-99.0,len(tabfits))
t_50_hci = []
t_q= np.repeat(-99.0,len(tabfits))
t_q_hci = []
log_ssfr= np.repeat(-99.0,len(tabfits))
log_ssfr_hci = []
for i in range(len(tabfits.index)):
    try:
        sfh = Rescale_sfh(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]])

        z_50[i] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        
        z_q[i] = sfh.z_q
        z_q_hci.append(sfh.z_q_hci)
        
        t_50[i] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        
        t_q[i] = sfh.t_q
        t_q_hci.append(sfh.t_q_hci)
        
        log_ssfr[i] = sfh.lssfr
        log_ssfr_hci.append(sfh.lssfr_hci)
        
    except:
        z_50_hci.append(np.array([0]))
        z_q_hci.append(np.array([0]))
        t_50_hci.append([0])
        t_q_hci.append([0])
        log_ssfr_hci.append([0])
        
print(z_50_hci)
tabfits['z_50'] = z_50
tabfits['z_50_hci'] = z_50_hci

tabfits['z_q'] = z_q
tabfits['z_q_hci'] = z_q_hci

tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci

tabfits['t_q'] = t_q
tabfits['t_q_hci'] = t_q_hci

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hci'] = log_ssfr_hci
    
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
compactness =[]

def A_value(Reff, mass):
    return (Reff) / (mass / 1E11)**0.75

for i in tabfits.index:
    A = A_value(tabfits.Reff[i], 10**tabfits.lmass[i])
    if A <= 1.5:
        compactness.append('u')
        
    if 1.5 < A <= 2.5:
        compactness.append('c')
        
    if A > 2.5:
        compactness.append('e')

tabfits['compact'] = compactness

tabfits.to_pickle('../dataframes/fitdb/tabfitdb.pkl')