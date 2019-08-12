import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_SF_sfh
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

select = pd.read_pickle('../Casey_data/SF_db.pkl')

#####parameters to get Z, m1-6, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {}
params = ['m', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'lwa']
k = ['Z', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lmass', 'Av', 'lwa']

for p in range(len(params)):
    m = np.repeat(-99.0,len(select))
    hci = []
    offmass = []

    for i in range(len(select.index)):
        try:
            x,px = np.load('../Casey_data/posteriors/{0}_{1}_SFfit_sim_p2_P{2}.npy'.format(select.field[select.index[i]], select.id[select.index[i]], params[p]))
            m[i], mreg, oreg = Highest_density_region(px,x)
            hci.append(mreg)
            offmass.append(oreg)
        except:
            hci.append([0])
            offmass.append([0])
    
    fitvals['{0}'.format(k[p])] = m
    fitvals['{0}_hci'.format(k[p])] = hci
    fitvals['{0}_modality'.format(k[p])] = offmass
    
#make db
tabfits = pd.DataFrame(fitvals)
tabfits['field'] = select.field.values
tabfits['id'] = select.id.values
tabfits['zgrism'] = select.zgrism.values
tabfits['AGN'] = select.AGN.values

#add SFH values
z_50= np.repeat(-99.0,len(tabfits))
z_50_hci = []
z_50_oreg = []

t_50= np.repeat(-99.0,len(tabfits))
t_50_hci = []
t_50_oreg = []

log_ssfr= np.repeat(-99.0,len(tabfits))
log_ssfr_hci = []
log_ssfr_oreg = []

for i in range(len(tabfits.index)):
    try:
        sfh = Rescale_SF_sfh(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]],tabfits.zgrism[tabfits.index[i]])

        z_50[i] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        z_50_oreg.append(sfh.z_50_offreg)

        t_50[i] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        t_50_oreg.append(sfh.t_50_offreg)
              
        log_ssfr[i] = sfh.lssfr
        log_ssfr_hci.append(sfh.lssfr_hci)
        log_ssfr_oreg.append(sfh.lssfr_offreg)
        
    except:
        z_50_hci.append(np.array([0]))
        t_50_hci.append([0])
        log_ssfr_hci.append([0])
        
        z_50_oreg.append(np.array([0]))
        t_50_oreg.append([0])
        log_ssfr_oreg.append([0])
        
        
tabfits['z_50'] = z_50
tabfits['z_50_hci'] = z_50_hci
tabfits['z_50_modality'] = z_50_hci

tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci
tabfits['t_50_modality'] = t_50_hci

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hci'] = log_ssfr_hci
tabfits['log_ssfr_modality'] = log_ssfr_oreg
    
tabfits.to_pickle('../Casey_data/SF_db_sim_p2.pkl')