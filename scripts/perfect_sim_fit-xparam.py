#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_exam import Gen_spec_2D
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from time import time
hpath = os.environ['HOME'] + '/'
    
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    logmass = float(sys.argv[4])
    trial = int(sys.argv[5])


dres = np.load(out_path + '{0}_{1}_{2}_psfit_p1.npy'.format(field, galaxy, trial),  allow_pickle = True).item() 

##save out P(z) and bestfit##

fit_dict = {}
params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm', 'z', 'd', 'bp1', 
          'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
P_params = ['Pm', 'Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm', 'Pz', 'Pd', 'Pbp1', 
          'Prp1', 'Pba', 'Pbb', 'Pbl', 'Pra', 'Prb', 'Prl']
bf_params = ['bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 'bfm7', 'bfm8', 'bfm9', 'bfm10', 'bflm', 'bfz', 'bfd', 'bfbp1', 
          'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']

bfits = dres.samples[-1]

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    fit_dict[params[i]] = t
    fit_dict[P_params[i]] = pt
    fit_dict[bf_params[i]] = bfits[i]

np.save(pos_path + '{0}_{1}_psfit_p1_fits_trial_{2}'.format(field, galaxy, trial),fit_dict)
