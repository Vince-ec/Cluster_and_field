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
        
###### create tabit db#########
list1 = glob(pos_path + '*Ifit*imp*.npy')

#####parameters to get Z, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {'Z':[], 'Z_hdr':[],'Z_modality':[],'t25':[], 't25_hdr':[],'t25_modality':[],
           't50':[], 't50_hdr':[],'t50_modality':[],'t75':[], 't75_hdr':[],'t75_modality':[],
           'log_ssfr':[], 'log_ssfr_hdr':[],'log_ssfr_modality':[], 'zgrism':[], 'zgrism_hdr':[],'zgrism_modality':[],
           'Av':[], 'Av_hdr':[],'Av_modality':[],'lmass':[], 'lmass_hdr':[],'lmass_modality':[], 
            'bfm':[], 'bft25':[], 'bft50':[], 'bft75':[], 'bflogssfr':[], 'bfz':[], 'bfd':[], 
          'bfbp1':[], 'bfrp1':[], 'bfba':[], 'bfbb':[], 'bfbl':[], 'bfra':[], 'bfrb':[], 'bfrl':[]}

k = ['Z','t25','t50','t75','log_ssfr','zgrism','Av','lmass', 'bfm', 'bft25', 'bft50', 'bft75', 'bflogssfr', 'bfz', 'bfd', 
          'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']

params = ['m', 't25', 't50', 't75', 'logssfr', 'z', 'd', 'lmass','bfm', 'bft25', 'bft50', 'bft75', 'bflogssfr', 'bfz', 'bfd', 
          'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
          
P_params = ['Pm','Pt25','Pt50','Pt75','Plogssfr','Pz','Pd','Plmass','-', '-', '-', '-', '-', '-', '-', 
          '-', '-', '-', '-', '-', '-', '-', '-']

for i in range(len(list1)):
    try:
        fit_db = np.load(pos_path + '{}_Ifit_impKIfits.npy'.format(i),allow_pickle=True).item()
        if len(fit_db)==47:
            for p in range(len(params)):
                if P_params[p] == '-':
                    fitvals[k[p]].append(fit_db[params[p]])
                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)
        else:
            for p in range(len(params)):
                if not params[p] in ['z', 'bfz']:
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
    except:
        for p in range(len(params)):
            if P_params[p] == '-':
                fitvals[k[p]].append(-99)
            else:
                fitvals['{}'.format(k[p])].append(-99)
                fitvals['{}_hdr'.format(k[p])].append([-99,-99])
                fitvals['{}_modality'.format(k[p])].append([-99])
        
#make db
for k in fitvals.keys():
    print(k, len(fitvals[k]))

tabfits = pd.DataFrame(fitvals)

#add SFH values
t_90= np.repeat(-99.0,len(tabfits))
t_90_hci = []
t_90_oreg = []
for i in range(len(tabfits.index)):
    try:
        with open(sfh_path + '{}_impKI_1D.pkl'.format(i), 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

        np.save(sfh_path + '{}_impKI'.format(i),[sfh.LBT, sfh.SFH],allow_pickle=True)
        np.save(sfh_path + '{}_impKI_16'.format(i),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
        np.save(sfh_path + '{}_impKI_84'.format(i),[sfh.LBT, sfh.SFH_84],allow_pickle=True)

        t_90[i] = sfh.t_90
        t_90_hci.append(sfh.t_90_hci)
        t_90_oreg.append(sfh.t_90_offreg)

    except:
        t_90_hci.append([-99])
        t_90_oreg.append([-99])
tabfits['t_90'] = t_90
tabfits['t_90_hdr'] = t_90_hci
tabfits['t_90_modality'] = t_90_oreg

tabfits.to_pickle(pos_path + 'Ifit_imp_KIdb.pkl')
