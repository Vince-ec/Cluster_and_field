#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
import dense_basis as db
from time import time
start = time()
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    
verbose=False
poolsize = 8

a = Oldest_galaxy(specz)
zscale = 0.035 * (1 + specz)

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019

    t25, t50, t75 = get_tx_vals([u[1],u[2],u[3]])
      
    logssfr = -15 + 8*u[4]
    
    z = stats.norm.ppf(u[5],loc = specz, scale = zscale)
    
    d = log_10_prior(u[6],[1E-3,2])
    
    bp1 = Gaussian_prior(u[7], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[8], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[9], [0.1,10])
    bb = log_10_prior(u[10], [0.0001,1])
    bl = log_10_prior(u[11], [0.01,1])
    
    ra = log_10_prior(u[12], [0.1,10])
    rb = log_10_prior(u[13], [0.0001,1])
    rl = log_10_prior(u[14], [0.01,1])
       
    return [m, t25, t50, t75, logssfr, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl]

def Galfit_L(X):
    m, t25, t50, t75, logssfr, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    sfh_tuple = np.hstack([0, logssfr, 3, t25,t50,t75])
    sfh, timeax = db.tuple_to_sfh(sfh_tuple, z)
    
    sp.set_tabular_sfh(timeax,sfh) 
    
    wave, flux = sp.get_spectrum(tage = timeax[-1], peraa = True)

    pmfl = Gs.Sim_phot_mult(wave * (1 + z),flux)
    
    SC = Scale_model(Gs.Pflx, Gs.Perr, pmfl)
    
    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux*SC, z, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
Gs = Gen_spec_2D(field, galaxy, specz, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                 phot_errterm = 0.04, irac_err = 0.08, mask = True)
####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 15, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)
dres = sampler.results
np.save(out_path + '{}_{}_KI'.format(field, galaxy), dres) 
  
end = time()
print(end - start)