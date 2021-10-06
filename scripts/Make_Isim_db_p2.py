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
from make_sfh_tool import Gen_sim_SFH

### set home for files
hpath = os.environ['HOME'] + '/'

#TODO: upload van der wel catalogs to cluster 

if hpath == '/home/vestrada78840/':
    data_path = '/scratch/user/vestrada78840/data/'
    pos_path = '/home/vestrada78840/posteriors/'
    sfh_path = '/scratch/user/vestrada78840/SFH/'
    phot_path = '/scratch/user/vestrada78840/phot/'
    db_path = '/scratch/user/vestrada78840/data/'
    
else:
    data_path = '../data/'
    pos_path = '../data/posteriors/'
    sfh_path = '../data/SFH/'
    phot_path = '../phot/'
    db_path = '../dataframes/galaxy_frames/'

#####parameters to get Z, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {'id':[],'Z':[], 'Z_hdr':[],'Z_modality':[],'lmass':[], 'lmass_hdr':[],
           'lmass_modality':[], 'Av':[], 'Av_hdr':[],'Av_modality':[],
           'zgrism':[], 'zgrism_hdr':[],'zgrism_modality':[],
           'bfm':[], 'bfa':[], 'bfm1':[], 
           'bfm2':[], 'bfm3':[], 'bfm4':[], 'bfm5':[], 'bfm6':[], 
           'bfm7':[], 'bfm8':[], 'bfm9':[], 'bfm10':[], 
            'bflm':[], 'bfd':[], 'bfbp1':[], 'bfrp1':[], 'bfba':[], 
           'bfbb':[], 'bfbl':[], 'bfra':[], 'bfrb':[], 'bfrl':[]}

k = ['Z','lmass', 'Av', 'zgrism','bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
     'bfm7', 'bfm8', 'bfm9', 'bfm10',       
     'bflm', 'bfd', 'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
params = ['m','lm', 'd','z','bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
          'bfm7', 'bfm8', 'bfm9', 'bfm10',  
            'bflm', 'bfd', 'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
P_params = ['Pm','Plm', 'Pd', 'Pz','-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 
            '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

###### create tabit db#########
for i in range(150):
    if os.path.isfile(pos_path + 'z1.0_{}_Ifit_impfits_p2.npy'.format(i)):
        fit_db = np.load(pos_path + 'z1.0_{}_Ifit_impfits_p2.npy'.format(i),allow_pickle=True).item()
        for p in range(len(params)):
            if not params[p] in ['z', 'bfm7', 'bfm8', 'bfm9', 'bfm10']:
                if P_params[p] == '-':
                    fitvals[k[p]].append(fit_db[params[p]])
                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)
            else:
                if params[p] == 'z':
                    fitvals['zgrism'].append(1.0)
                    fitvals['zgrism_hdr'].append([-99,-99])
                    fitvals['zgrism_modality'].append([-99])
                else:                    
                    fitvals[k[p]].append(-99)

    else:
        for p in range(len(params)):
            if P_params[p] == '-':
                fitvals[k[p]].append(-99)
            else:
                fitvals['{}'.format(k[p])].append(-99)
                fitvals['{}_hdr'.format(k[p])].append(-99)
                fitvals['{}_modality'.format(k[p])].append(-99)
                    
                    
                    
for i in range(150):
    if os.path.isfile(pos_path + 'z1.5_{}_Ifit_impfits_p2.npy'.format(i)):
        fit_db = np.load(pos_path + 'z1.5_{}_Ifit_impfits_p2.npy'.format(i),allow_pickle=True).item()
        for p in range(len(params)):
            if not params[p] in ['z', 'bfm7', 'bfm8', 'bfm9', 'bfm10']:
                if P_params[p] == '-':
                    fitvals[k[p]].append(fit_db[params[p]])
                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)
            else:
                if params[p] == 'z':
                    fitvals['zgrism'].append(1.5)
                    fitvals['zgrism_hdr'].append([-99,-99])
                    fitvals['zgrism_modality'].append([-99])
                else:                    
                    fitvals[k[p]].append(-99)

    else:
        for p in range(len(params)):
            if P_params[p] == '-':
                fitvals[k[p]].append(-99)
            else:
                fitvals['{}'.format(k[p])].append(-99)
                fitvals['{}_hdr'.format(k[p])].append(-99)
                fitvals['{}_modality'.format(k[p])].append(-99)
                    


for i in range(135):
    if os.path.isfile(pos_path + 'z2.0_{}_Ifit_impfits_p2.npy'.format(i)):
        fit_db = np.load(pos_path + 'z2.0_{}_Ifit_impfits_p2.npy'.format(i),allow_pickle=True).item()
        for p in range(len(params)):
            if not params[p] in ['z', 'bfm7', 'bfm8', 'bfm9', 'bfm10']:
                if P_params[p] == '-':
                    fitvals[k[p]].append(fit_db[params[p]])
                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)
            else:
                if params[p] == 'z':
                    fitvals['zgrism'].append(2.0)
                    fitvals['zgrism_hdr'].append([-99,-99])
                    fitvals['zgrism_modality'].append([-99])
                else:                    
                    fitvals[k[p]].append(-99)

    else:
        for p in range(len(params)):
            if P_params[p] == '-':
                fitvals[k[p]].append(-99)
            else:
                fitvals['{}'.format(k[p])].append(-99)
                fitvals['{}_hdr'.format(k[p])].append(-99)
                fitvals['{}_modality'.format(k[p])].append(-99)
                    

                    
#make db
for k in fitvals.keys():
    print(k, len(fitvals[k]))

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

for i in range(150):
    if os.path.isfile(pos_path + 'z1.0_{}_Ifit_impfits_p2.npy'.format(i)):
        with open(glob(sfh_path + 'z1.0_{}_imp_p2_1D.pkl'.format(i))[0], 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

        np.save(sfh_path + 'z1.0_{}_imp_p2'.format(i),[sfh.LBT, sfh.SFH],allow_pickle=True)
        np.save(sfh_path + 'z1.0_{}_imp_p2_16'.format(i),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
        np.save(sfh_path + 'z1.0_{}_imp_p2_84'.format(i),[sfh.LBT, sfh.SFH_84],allow_pickle=True)

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
    else:
        z_50_hci.append([-99])
        z_50_oreg.append([-99])

        z_80_hci.append([-99])
        z_80_oreg.append([-99])

        z_90_hci.append([-99])
        z_90_oreg.append([-99])

        t_50_hci.append([-99])
        t_50_oreg.append([-99])

        t_80_hci.append([-99])
        t_80_oreg.append([-99])

        t_90_hci.append([-99])
        t_90_oreg.append([-99])

        log_ssfr_hci.append([-99])
        log_ssfr_oreg.append([-99])

for i in range(150):
    if os.path.isfile(pos_path + 'z1.5_{}_Ifit_impfits_p2.npy'.format(i)):
        with open(glob(sfh_path + 'z1.5_{}_imp_p2_1D.pkl'.format(i))[0], 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

        np.save(sfh_path + 'z1.5_{}_imp_p2'.format(i),[sfh.LBT, sfh.SFH],allow_pickle=True)
        np.save(sfh_path + 'z1.5_{}_imp_p2_16'.format(i),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
        np.save(sfh_path + 'z1.5_{}_imp_p2_84'.format(i),[sfh.LBT, sfh.SFH_84],allow_pickle=True)
            
        z_50[i + 150] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        z_50_oreg.append(sfh.z_50_offreg)

        z_80[i + 150] = sfh.z_80
        z_80_hci.append(sfh.z_80_hci)
        z_80_oreg.append(sfh.z_80_offreg)

        z_90[i + 150] = sfh.z_90
        z_90_hci.append(sfh.z_90_hci)
        z_90_oreg.append(sfh.z_90_offreg)

        t_50[i + 150] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        t_50_oreg.append(sfh.t_50_offreg)

        t_80[i + 150] = sfh.t_80
        t_80_hci.append(sfh.t_80_hci)
        t_80_oreg.append(sfh.t_80_offreg)

        t_90[i + 150] = sfh.t_90
        t_90_hci.append(sfh.t_90_hci)
        t_90_oreg.append(sfh.t_90_offreg)

        log_ssfr[i + 150] =  np.log10((np.trapz(sfh.SFH[:11], sfh.LBT[:11])/0.1) / 10**tabfits.lmass[i + 150])
        log_ssfr_hci.append(sfh.lssfr_hci)
        log_ssfr_oreg.append(sfh.lssfr_offreg)
    else:
        z_50_hci.append([-99])
        z_50_oreg.append([-99])

        z_80_hci.append([-99])
        z_80_oreg.append([-99])

        z_90_hci.append([-99])
        z_90_oreg.append([-99])

        t_50_hci.append([-99])
        t_50_oreg.append([-99])

        t_80_hci.append([-99])
        t_80_oreg.append([-99])

        t_90_hci.append([-99])
        t_90_oreg.append([-99])

        log_ssfr_hci.append([-99])
        log_ssfr_oreg.append([-99]) 

for i in range(135):
    if os.path.isfile(pos_path + 'z2.0_{}_Ifit_impfits_p2.npy'.format(i)):
        with open(glob(sfh_path + 'z2.0_{}_imp_p2_1D.pkl'.format(i))[0], 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

        np.save(sfh_path + 'z2.0_{}_imp_p2'.format(i),[sfh.LBT, sfh.SFH],allow_pickle=True)
        np.save(sfh_path + 'z2.0_{}_imp_p2_16'.format(i),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
        np.save(sfh_path + 'z2.0_{}_imp_p2_84'.format(i),[sfh.LBT, sfh.SFH_84],allow_pickle=True)
            
        z_50[i + 150 + 150] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        z_50_oreg.append(sfh.z_50_offreg)

        z_80[i + 150 + 150] = sfh.z_80
        z_80_hci.append(sfh.z_80_hci)
        z_80_oreg.append(sfh.z_80_offreg)

        z_90[i + 150 + 150] = sfh.z_90
        z_90_hci.append(sfh.z_90_hci)
        z_90_oreg.append(sfh.z_90_offreg)

        t_50[i + 150 + 150] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        t_50_oreg.append(sfh.t_50_offreg)

        t_80[i + 150 + 150] = sfh.t_80
        t_80_hci.append(sfh.t_80_hci)
        t_80_oreg.append(sfh.t_80_offreg)

        t_90[i + 150 + 150] = sfh.t_90
        t_90_hci.append(sfh.t_90_hci)
        t_90_oreg.append(sfh.t_90_offreg)

        log_ssfr[i + 150 + 150] =  np.log10((np.trapz(sfh.SFH[:11], sfh.LBT[:11])/0.1) / 10**tabfits.lmass[i + 150 + 150] )
        log_ssfr_hci.append(sfh.lssfr_hci)
        log_ssfr_oreg.append(sfh.lssfr_offreg)
    else:
        z_50_hci.append([-99])
        z_50_oreg.append([-99])

        z_80_hci.append([-99])
        z_80_oreg.append([-99])

        z_90_hci.append([-99])
        z_90_oreg.append([-99])

        t_50_hci.append([-99])
        t_50_oreg.append([-99])

        t_80_hci.append([-99])
        t_80_oreg.append([-99])

        t_90_hci.append([-99])
        t_90_oreg.append([-99])

        log_ssfr_hci.append([-99])
        log_ssfr_oreg.append([-99]) 
    
    
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
   
tabfits.to_pickle(pos_path + 'Ifit_imp_p2_db.pkl')
