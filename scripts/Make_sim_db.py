import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_sfh
from glob import glob
import os
import pickle
from make_sfh_tool import Gen_SFH

### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/scratch/user/vestrada78840/data/'
    pos_path = '/home/vestrada78840/posteriors/'
    sfh_path = '/scratch/user/vestrada78840/SFH/'
    phot_path = '/scratch/user/vestrada78840/phot/'
    db_path = '/scratch/user/vestrada78840/data/'
    ### set home for files
    
else:
    data_path = '../data/'
    pos_path = '../data/posteriors/'
    sfh_path = '../data/SFH/'
    phot_path = '../phot/'
    db_path = '../dataframes/galaxy_frames/'
    
#make a dictionary
fitvals = {'Z':[], 'Z_hdr':[],'Z_modality':[],'lmass':[], 'lmass_hdr':[],'lmass_modality':[], 
           'Av':[], 'Av_hdr':[],'Av_modality':[],'lwa':[],'lwa_hdr':[], 'lwa_modality':[],
            'z':[],'z_hdr':[], 'z_modality':[],'bfm':[], 'bfa':[], 'bfm1':[], 
           'bfm2':[], 'bfm3':[], 'bfm4':[], 'bfm5':[], 'bfm6':[], 'bfm7':[], 'bfm8':[], 'bfm9':[], 'bfm10':[], 
            'bflm':[], 'bfd':[], 'bfz':[],'bfbp1':[], 'bfrp1':[], 'bfba':[], 
           'bfbb':[], 'bfbl':[], 'bfra':[], 'bfrb':[], 'bfrl':[]}

k = ['Z','lmass', 'Av','lwa', 'zgrism', 'bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
    'bfm7', 'bfm8', 'bfm9', 'bfm10', 'bflm', 'bfd', 'bfz', 'bfbp1', 'bfrp1', 'bfba', 
    'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
params = ['m','lm', 'd','lwa', 'z', 'bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
        'bfm7', 'bfm8', 'bfm9', 'bfm10', 'bflm', 'bfd', 'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 
        'bfra', 'bfrb', 'bfrl']
P_params = ['Pm','Plm', 'Pd', 'Plwa', 'Pz', '-', '-', '-', '-', '-', '-', '-', 
        '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 
        '-', '-', '-']

for i in range(100):
    fit_db = np.load(pos_path + 'GSD_39170_psfit_p1_fits_trial_{}.npy'.format(i + 1),allow_pickle=True).item()
    for p in range(len(params)):
        if P_params[p] == '-':
            fitvals[k[p]].append(fit_db[params[p]])
        else:
            m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
            fitvals['{}'.format(k[p])].append(m)
            fitvals['{}_hdr'.format(k[p])].append(mreg)
            fitvals['{}_modality'.format(k[p])].append(oreg)
                
tabfits = pd.DataFrame(fitvals)

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
t_50= np.repeat(-99.0,len(tabfits))
t_50_hci = []
t_50_oreg = []
t_80= np.repeat(-99.0,len(tabfits))
t_80_hci = []
t_80_oreg = []
t_90= np.repeat(-99.0,len(tabfits))
t_90_hci = []
t_90_oreg = []
log_ssfr= np.repeat(-99.0,len(tabfits))
log_ssfr_hci = []
log_ssfr_oreg = []
for i in range(100):
    with open(sfh_path + 'GSD_39170_psfit_p1_fits_trial_{}.npy_1D.pkl'.format(i+1), 'rb') as sfh_file:
        sfh = pickle.load(sfh_file)
    z_50[i] = sfh.z_50
    z_50_hci.append(sfh.z_50_hci)
    z_50_oreg.append(sfh.z_50_offreg)

    z_80[i] = sfh.z_80
    z_80_hci.append(sfh.z_80_hci)
    z_80_oreg.append(sfh.z_80_offreg)

    z_90[i] = sfh.z_90
    z_90_hci.append(sfh.z_90_hci)
    z_90_oreg.append(sfh.z_90_offreg)

    t_50[i] = sfh.t_50
    t_50_hci.append(sfh.t_50_hci)
    t_50_oreg.append(sfh.t_50_offreg)

    t_80[i] = sfh.t_80
    t_80_hci.append(sfh.t_80_hci)
    t_80_oreg.append(sfh.t_80_offreg)

    t_90[i] = sfh.t_90
    t_90_hci.append(sfh.t_90_hci)
    t_90_oreg.append(sfh.t_90_offreg)

    log_ssfr[i] = np.log10((np.trapz(sfh.SFH[:11], sfh.LBT[:11])/0.1) / 10**tabfits.lmass[i])
    log_ssfr_hci.append(sfh.lssfr_hci)
    log_ssfr_oreg.append(sfh.lssfr_offreg)

tabfits['z_50'] = z_50
tabfits['z_50_hdr'] = z_50_hci
tabfits['z_50_modality'] = z_50_oreg

tabfits['z_80'] = z_80
tabfits['z_80_hdr'] = z_80_hci
tabfits['z_80_modality'] = z_80_oreg

tabfits['z_90'] = z_90
tabfits['z_90_hdr'] = z_90_hci
tabfits['z_90_modality'] = z_90_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hdr'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_80'] = t_80
tabfits['t_80_hdr'] = t_80_hci
tabfits['t_80_modality'] = t_80_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hdr'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_90'] = t_90
tabfits['t_90_hdr'] = t_90_hci
tabfits['t_90_modality'] = t_90_oreg

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hdr'] = log_ssfr_hci
tabfits['log_ssfr_modality'] = log_ssfr_oreg
   

tabfits.to_pickle(pos_path + 'simfits.pkl')
