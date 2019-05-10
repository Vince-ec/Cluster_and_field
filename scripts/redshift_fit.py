#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    photz = float(sys.argv[3])
    
verbose=True
poolsize = 8

def zfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    
    a = (3)* u[1] + 1
    
    t = Gaussian_prior(u[2], [0.01,2],0,.5)

    lm = Gaussian_prior(u[3], [9.5, 12.5], 11, 0.75)
     
    z = Gaussian_prior(u[4], [photz - 0.3, photz + 0.3], photz, 0.15)
      
    bsc= Gaussian_prior(u[5], [0.8, 1.2], 1, 0.05)
    rsc= Gaussian_prior(u[6], [0.8, 1.2], 1, 0.05)
    bp1 = Gaussian_prior(u[7], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[8], [-0.05,0.05], 0, 0.025)
   
    return [m, a, t, lm, z, bsc, rsc, bp1, rp1]

        
def zfit_L(X):
    m, a, t, lm, z, bsc, rsc, bp1, rp1 = X

    sp.params['logzsol'] = np.log10(m)
    sp.params['tau'] = t
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate(Gmfl, [bp1, rp1], [bsc, rsc], wvs)
    
    Gchi, Pchi = Full_fit(Gs, Gmfl, Pmfl, wvs, flxs, errs)

    return -0.5 * (Gchi+Pchi)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau=0.1)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, phot_errterm = 0.02, irac_err = 0.04) 

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.NestedSampler(zfit_L, zfit_prior, ndim = 9, sample = 'rwalk', bound = 'single',
                                    pool=Pool(processes= poolsize), queue_size = poolsize, nlive_points = 2000)

sampler.run_nested(print_progress = verbose)

dres = sampler.results

np.save(out_path + '{0}_{1}_zfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

t,pt = Get_posterior(dres,4)
np.save(pos_path + '{0}_{1}_zfit_Pz'.format(field, galaxy),[t,pt])

m, a, t, lm, z, bsc, rsc, bp1, rp1 = dres.samples[-1]

np.save(pos_path + '{0}_{1}_zfit_bfit'.format(field, galaxy), 
        [m, a, t, lm, z, bsc, rsc, bp1, rp1, dres.logl[-1]])