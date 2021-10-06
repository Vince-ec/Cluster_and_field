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
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])

Gs = Gen_spec(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 

dres = np.load(out_path + '{}_{}_KI.npy'.format(field, galaxy),  allow_pickle = True).item() 

##save out P(z) and bestfit##

fit_dict = {}
params = ['m', 't25', 't50', 't75', 'logssfr', 'z', 'd', 
          'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']

P_params = ['Pm', 'Pt25', 'Pt50', 'Pt75', 'Plogssfr', 'Pz', 'Pd', 
          'Pbp1', 'Prp1', 'Pba', 'Pbb', 'Pbl', 'Pra', 'Prb', 'Prl']

bf_params = ['bfm', 'bft25', 'bft50', 'bft75', 'bflogssfr', 'bfz', 'bfd', 
          'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']

bfits = dres.samples[-1]

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    fit_dict[params[i]] = t
    fit_dict[P_params[i]] = pt
    fit_dict[bf_params[i]] = bfits[i]

np.save(pos_path + '{}_{}_KI'.format(field, galaxy),fit_dict)

sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    
def get_lwa_SF_u(params, agebins,sp):
    m, a, m1, m2, m3, m4, m5, m6 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_u'])
    sp.params['compute_light_ages'] = False
    
    return lwa

    
lwa_g = []
lwa_u = []
lwa_r = []
mass = []
for i in range(len(dres.samples)):
    m, t25, t50, t75, logssfr, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl = dres.samples[i]
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    sfh_tuple = np.hstack([0, logssfr, 3, t25,t50,t75])
    sfh, timeax = db.tuple_to_sfh(sfh_tuple, z)
    
    sp.set_tabular_sfh(timeax,sfh) 
    
    sp.params['compute_light_ages'] = True
    lwa_g.append(sp.get_mags(tage = timeax[-1], bands=['sdss_g'])[0])
    lwa_u.append(sp.get_mags(tage = timeax[-1], bands=['sdss_u'])[0])
    lwa_r.append(sp.get_mags(tage = timeax[-1], bands=['sdss_r'])[0])
    sp.params['compute_light_ages'] = False
    
    wave, flux = sp.get_spectrum(tage = timeax[-1], peraa = True)
    pmfl = Gs.Sim_phot_mult(wave * (1 + z),F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass))
    
    mass.append(Scale_model(Gs.Pflx, Gs.Perr, pmfl))

g,pg = Get_derived_posterior(np.array(lwa_g), dres)
u,pu = Get_derived_posterior(np.array(lwa_u), dres)
r,pr = Get_derived_posterior(np.array(lwa_r), dres)
lm, plm = Get_derived_posterior(np.log10(mass), dres)

fit_db = np.load(pos_path + '{}_{}_KI.npy'.format(field, galaxy), allow_pickle=True).item()

fit_db['Plwa_g'] = pg
fit_db['lwa_g'] = g

fit_db['Plwa_u'] = pu
fit_db['lwa_u'] = u

fit_db['Plwa_r'] = pr
fit_db['lwa_r'] = r

fit_db['Plmass'] = plm
fit_db['lmass'] = lm

np.save(pos_path + '{}_{}_KI'.format(field, galaxy),fit_db)