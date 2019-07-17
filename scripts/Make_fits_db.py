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
    offmass = []
    
    for i in range(len(select.index)):
        try:
            x,px = np.load('../data/posteriors/{0}_{1}_tabfit_P{2}.npy'.format(select.field[select.index[i]], select.id[select.index[i]], params[p]))
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

#add SFH values
z_50= np.repeat(-99.0,len(tabfits))
z_50_hci = []
z_50_oreg = []
z_80= np.repeat(-99.0,len(tabfits))
z_80_hci = []
z_80_oreg = []
z_90= np.repeat(-99.0,len(tabfits))
z_90_hci = []
z_90_oreg = []
z_q= np.repeat(-99.0,len(tabfits)) 
z_q_hci = []
z_q_oreg = []
t_50= np.repeat(-99.0,len(tabfits))
t_50_hci = []
t_50_oreg = []
t_80= np.repeat(-99.0,len(tabfits))
t_80_hci = []
t_80_oreg = []
t_90= np.repeat(-99.0,len(tabfits))
t_90_hci = []
t_90_oreg = []
t_q= np.repeat(-99.0,len(tabfits))
t_q_hci = []
t_q_oreg = []
log_ssfr= np.repeat(-99.0,len(tabfits))
log_ssfr_hci = []
log_ssfr_oreg = []
for i in range(len(tabfits.index)):
    try:
        sfh = Rescale_sfh(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]],5000)
        np.save('../data/SFH/{}_{}.npy'.format(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]]), 
                [sfh.LBT, sfh.SFH])
        print(sfh.z_50)
        z_50[i] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        z_50_oreg.append(sfh.z_50_offreg)

        z_80[i] = sfh.z_80
        z_80_hci.append(sfh.z_80_hci)
        z_80_oreg.append(sfh.z_80_offreg)

        z_90[i] = sfh.z_90
        z_90_hci.append(sfh.z_90_hci)
        z_90_oreg.append(sfh.z_90_offreg)

        z_q[i] = sfh.z_q
        z_q_hci.append(sfh.z_q_hci)
        z_q_oreg.append(sfh.z_q_offreg)

        t_50[i] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        t_50_oreg.append(sfh.t_50_offreg)

        t_80[i] = sfh.t_80
        t_80_hci.append(sfh.t_80_hci)
        t_80_oreg.append(sfh.t_80_offreg)

        t_90[i] = sfh.t_90
        t_90_hci.append(sfh.t_90_hci)
        t_90_oreg.append(sfh.t_90_offreg)

        t_q[i] = sfh.t_q
        t_q_hci.append(sfh.t_q_hci)
        t_q_oreg.append(sfh.t_q_offreg)

        log_ssfr[i] = sfh.lssfr
        log_ssfr_hci.append(sfh.lssfr_hci)
        log_ssfr_oreg.append(sfh.lssfr_offreg)
        
    except:
        z_50_hci.append(np.array([0]))
        z_80_hci.append(np.array([0]))
        z_90_hci.append(np.array([0]))
        z_q_hci.append(np.array([0]))
        t_50_hci.append([0])
        t_80_hci.append([0])
        t_90_hci.append([0])
        t_q_hci.append([0])
        log_ssfr_hci.append([0])
        
        z_50_oreg.append(np.array([0]))
        z_80_oreg.append(np.array([0]))
        z_90_oreg.append(np.array([0]))
        z_q_oreg.append(np.array([0]))
        t_50_oreg.append([0])
        t_80_oreg.append([0])
        t_90_oreg.append([0])
        t_q_oreg.append([0])
        log_ssfr_oreg.append([0])
        
        
tabfits['z_50'] = z_50
tabfits['z_50_hci'] = z_50_hci
tabfits['z_50_modality'] = z_50_oreg

tabfits['z_80'] = z_80
tabfits['z_80_hci'] = z_80_hci
tabfits['z_80_modality'] = z_80_oreg

tabfits['z_90'] = z_90
tabfits['z_90_hci'] = z_90_hci
tabfits['z_90_modality'] = z_90_oreg


tabfits['z_q'] = z_q
tabfits['z_q_hci'] = z_q_hci
tabfits['z_q_modality'] = z_q_oreg


tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_80'] = t_80
tabfits['t_80_hci'] = t_80_hci
tabfits['t_80_modality'] = t_80_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_90'] = t_90
tabfits['t_90_hci'] = t_90_hci
tabfits['t_90_modality'] = t_90_oreg

tabfits['t_q'] = t_q
tabfits['t_q_hci'] = t_q_hci
tabfits['t_q_modality'] = t_q_oreg


tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hci'] = log_ssfr_hci
tabfits['log_ssfr_modality'] = log_ssfr_oreg

tabfits.to_pickle('../dataframes/fitdb/fullfitdb.pkl')