#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
from spec_exam import Gen_spec_2D
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from time import time
import dense_basis as db

hpath = os.environ['HOME'] + '/'
    
if __name__ == '__main__':
    galaxy = int(sys.argv[1])
    lwa = float(sys.argv[2])

dres = np.load(out_path + '{}_Ifit_impKI.npy'.format(galaxy),  allow_pickle = True).item() 

##save out P(z) and bestfit##
fit_dict = {}

if lwa > 1:
    params = ['m', 't25', 't50', 't75', 'logssfr', 'z', 'd', 
              'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']

    P_params = ['Pm', 'Pt25', 'Pt50', 'Pt75', 'Plogssfr', 'Pz', 'Pd', 
              'Pbp1', 'Prp1', 'Pba', 'Pbb', 'Pbl', 'Pra', 'Prb', 'Prl']

    bf_params = ['bfm', 'bft25', 'bft50', 'bft75', 'bflogssfr', 'bfz', 'bfd', 
              'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']

else:
    params = ['m', 't25', 't50', 't75', 'logssfr', 'd', 
              'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']

    P_params = ['Pm', 'Pt25', 'Pt50', 'Pt75', 'Plogssfr', 'Pd', 
              'Pbp1', 'Prp1', 'Pba', 'Pbb', 'Pbl', 'Pra', 'Prb', 'Prl']

    bf_params = ['bfm', 'bft25', 'bft50', 'bft75', 'bflogssfr', 'bfd', 
              'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
    
bfits = dres.samples[-1]

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    fit_dict[params[i]] = t
    fit_dict[P_params[i]] = pt
    fit_dict[bf_params[i]] = bfits[i]

np.save(pos_path + '{}_Ifit_impKIfits'.format(galaxy),fit_dict)


if lwa > 1:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    Gs = Gen_spec_2D('GSD', 39170, 1.5, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                 phot_errterm = 0.04, irac_err = 0.08, mask = False)
else:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    Gs = Gen_spec_2D('GND',27930, 1.5, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                     phot_errterm = 0.04, irac_err = 0.08, mask = True)    
mass = []
for i in range(len(dres.samples)):
    if lwa > 1:
        m, t25, t50, t75, logssfr, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl = dres.samples[i]
    else:
        m, t25, t50, t75, logssfr, d, bp1, rp1, ba, bb, bl, ra, rb, rl = dres.samples[i]
        z=1.5
        
    sp.params['dust2'] = d
    if lwa > 1:
        sp.params['dust1'] = d
        
    sp.params['logzsol'] = np.log10(m)
    
    sfh_tuple = np.hstack([0, logssfr, 3, t25,t50,t75])
    sfh, timeax = tuple_to_sfh_stand_alone(sfh_tuple, z)
    
    sp.set_tabular_sfh(timeax,sfh) 
    
    wave, flux = sp.get_spectrum(tage = timeax[-1], peraa = True)
    pmfl = Gs.Sim_phot_mult(wave * (1 + z),F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass))
    
    mass.append(Scale_model(Gs.Pflx, Gs.Perr, pmfl))

lm, plm = Get_derived_posterior(np.log10(mass), dres)

fit_db = np.load(pos_path + '{}_Ifit_impKIfits.npy'.format(galaxy), allow_pickle=True).item()

fit_db['Plmass'] = plm
fit_db['lmass'] = lm

np.save(pos_path + '{}_Ifit_impKIfits'.format(galaxy),fit_db)
