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
    specz = float(sys.argv[3])
    
verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
zscale = 0.035 * (1 + specz)

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    
    a = (agelim - 1)* u[1] + 1

    t = Gaussian_prior(u[2], [0.01,2],0,.5)

    lm = Gaussian_prior(u[3], [9.5, 12.5], 11, 0.75)
     
    z = stats.norm.ppf(u[4],loc = specz, scale = zscale)
   
    d = log_10_prior(u[5],[1E-3,2])
    
    bsc= Gaussian_prior(u[6], [0.8, 1.2], 1, 0.05)
    rsc= Gaussian_prior(u[7], [0.8, 1.2], 1, 0.05)
    bp1 = Gaussian_prior(u[8], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[9], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[10], [0.1,10])
    bb = log_10_prior(u[11], [0.0001,1])
    bl = log_10_prior(u[12], [0.01,1])
    
    ra = log_10_prior(u[13], [0.1,10])
    rb = log_10_prior(u[14], [0.0001,1])
    rl = log_10_prior(u[15], [0.01,1])
   
    lwa = get_lwa_delay([m, a, t], get_agebins(a),sp)[0]
    
    return [m, a, t, lm, z, d, bsc, rsc, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa]

def Galfit_L(X):
    m, a, t, lm, z, d, bsc, rsc, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    sp.params['tau'] = t
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate(Gmfl, [bp1, rp1], [bsc, rsc], wvs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)


#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau=0.1, dust_type = 1)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = False,
        phot_errterm = 0.04, irac_err = 0.08, decontam = True) 

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 17, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_delayfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 't', 'lm', 'z', 'd', 'bsc', 'rsc', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_delayfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bft, bflm, bfz, bfd, bfbsc, bfrsc, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa= dres.samples[-1]

np.save(pos_path + '{0}_{1}_delayfit_bfit'.format(field, galaxy),
        [bfm, bfa, bft, bflm, bfz, bfd, bfbsc, bfrsc, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa, dres.logl[-1]])
