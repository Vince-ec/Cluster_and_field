import pandas as pd
import pickle
from astropy.io import fits
from astropy.table import Table
import re
from spec_tools import Gen_PPF, boot_to_posterior, convert_sfh, Derive_SFH_weights, Highest_density_region
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import astropy.units as u
from make_sfh_tool import Gen_sim_SFH
### set home for files
hpath = os.environ['HOME'] + '/'

sfh_path = '/scratch/user/vestrada78840/SFH/'
pos_path = '/home/vestrada78840/posteriors/'


if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    sfa = sys.argv[4]

if sfa == 'Q':
    fname = '{}_{}_tabfit.npy'.format(field,galaxy)

    sfh = Gen_sim_SFH(fname, 5000, specz)
    
    with open(sfh_path + '{}_{}_1D.pkl'.format(field, galaxy), 'wb') as output:
        pickle.dump(sfh, output, pickle.HIGHEST_PROTOCOL)
        
else:
    fname ='{}_{}_SFfit_p1_fits.npy'.format(field,galaxy)

    sfh = Gen_sim_SFH(fname, 5000, specz)

    with open(sfh_path + '{}_{}_p1_1D.pkl'.format(field, galaxy), 'wb') as output:
        pickle.dump(sfh, output, pickle.HIGHEST_PROTOCOL)

fit_db = np.load(pos_path + fname, allow_pickle=True).item()

fit_db['Pssfr'] = sfh.Pssfr
fit_db['ssfr'] = sfh.ssfr

np.save(pos_path + fname,fit_db, allow_pickle=True)
        

np.save(sfh_path + '{}_{}'.format(field, galaxy),[sfh.LBT, sfh.SFH],allow_pickle=True)
np.save(sfh_path + '{}_{}_16'.format(field, galaxy),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
np.save(sfh_path + '{}_{}_84'.format(field, galaxy),[sfh.LBT, sfh.SFH_84],allow_pickle=True)