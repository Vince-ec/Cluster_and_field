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

verbose=True
poolsize = 8

def zfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    
    a = (7)* u[1] + 1
    
    z = 3 * u[2]
        
    bsc= Gaussian_prior(u[3], [0.8, 1.2], 1, 0.05)
    rsc= Gaussian_prior(u[4], [0.8, 1.2], 1, 0.05)
    bp1 = Gaussian_prior(u[5], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[6], [-0.05,0.05], 0, 0.025)
    
    return [m, a, z, bsc, rsc, bp1, rp1]
        
def zfit_L(X):
    m, a, z, bsc, rsc, bp1, rp1 = X

    sp.params['logzsol'] = np.log10(m)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z, wvs, flxs, errs, beams, trans)
              
    PC= Full_scale(Gs, Pmfl)

    Gmfl = Full_calibrate(PC* Gmfl, [bp1, rp1], [bsc, rsc], wvs)
    
    
    Gchi, Pchi = Full_fit(Gs, Gmfl, PC*Pmfl, wvs, flxs, errs)
                  
    return -0.5 * (Gchi + Pchi)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1),sfh = 0)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, g102_lims=[8300, 11288], g141_lims=[11288, 16500], mdl_err = False,
        phot_errterm = 0.04, irac_err = 0.08, decontam = True) 

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.NestedSampler(zfit_L, zfit_prior, ndim = 7, sample = 'rwalk', bound = 'multi',
                                    pool=Pool(processes= poolsize), queue_size = poolsize)

sampler.run_nested(print_progress = verbose)

dres = sampler.results

np.save(out_path + '{0}_{1}_zfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

t,pt = Get_posterior(dres,2)
np.save(pos_path + '{0}_{1}_zfit_Pz'.format(field, galaxy),[t,pt])

bfm, bfa, bfz, bfbsc, bfrsc, bfbp1, bfrp1 = dres.samples[-1]

np.save(pos_path + '{0}_{1}_zfit_bfit'.format(field, galaxy), [bfm, bfa, bfz, bfbsc, bfrsc, bfbp1, bfrp1, dres.logl[-1]])