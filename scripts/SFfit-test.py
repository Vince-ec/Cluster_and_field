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
    logmass = float(sys.argv[4])
    
verbose=False
poolsize = 8

def Z_prior_mu(lmass):
    M = [9, 11.5]
    P = [-0.5813, 0.06818]
    iP = interp1d(M,P)
    return iP(lmass) 

onesig = (0.04 + 0.47)/2

#if (Z_prior_mu(logmass) - 2*onesig) < np.log10(0.001 / 0.019):
mllim = np.log10(0.001 / 0.019)
#else:
#    mllim = Z_prior_mu(logmass) - 2*onesig
    
#if (Z_prior_mu(logmass) + 2*onesig) > np.log10(0.03 / 0.019):
mhlim = np.log10(0.031 / 0.019)
#else:
#    mhlim = Z_prior_mu(logmass) + 2*onesig

agelim = Oldest_galaxy(specz)

def Galfit_prior(u):
    m = 10**Gaussian_prior(u[0], [mllim, mhlim], Z_prior_mu(logmass), 2*onesig)
    a = (agelim - 1)* u[1] + 1
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))
    
    lm = Gaussian_prior(u[8], [8.0, 12.5], 11, 0.75)
     
    d = 4*u[9]
   
    lwa = get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum = 6),sp)[0]
    
    return [m, a, m1, m2, m3, m4, m5, m6, lm, d, lwa]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, lm, d, lwa = X
    
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Pmfl = Gs.Sim_phot_mult(wave * (1 + specz),
                            F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm)
     
    return  np.sum((((Gs.Pflx - Pmfl) / Gs.Perr)**2))

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

###########gen spec##########
Gs = Gen_SF_spec(field, galaxy, 1, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
        phot_errterm = 0.04, irac_err = 0.08) 

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 11, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_SFphotfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##
params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'lwa']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFphotfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd,blwa = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFphotfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, blwa, dres.logl[-1]])