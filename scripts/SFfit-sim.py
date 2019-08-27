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
    Av = float(sys.argv[5])
    Z = float(sys.argv[6])
    a = float(sys.argv[7])
    m1 = float(sys.argv[8])
    m2 = float(sys.argv[9])
    m3 = float(sys.argv[10])
    m4 = float(sys.argv[11])
    m5 = float(sys.argv[12])
    m6 = float(sys.argv[13])

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
#a = 3

def Galfit_prior(u):
    m = 10**Gaussian_prior(u[0], [mllim, mhlim], Z_prior_mu(logmass), onesig)
    a = (agelim - 1)* u[1] + 1
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))
    
    lm = Gaussian_prior(u[8], [8.0, 12.5], 11, 0.75)
     
    d = 4*u[9]
    
    bp1 = Gaussian_prior(u[10], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[11], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[12], [0.1,10])
    bb = log_10_prior(u[13], [0.0001,1])
    bl = log_10_prior(u[14], [0.01,1])
    
    ra = log_10_prior(u[15], [0.1,10])
    rb = log_10_prior(u[16], [0.0001,1])
    rl = log_10_prior(u[17], [0.01,1])
   
    lwa = get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum = 6),sp)[0]
    
    return [m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa = X
    
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm, specz, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_sim(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
sp.params['dust2'] = Av
sp.params['logzsol'] = Z

time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), 
    [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

sp.set_tabular_sfh(time,sfr) 

wv,fl = sp.get_spectrum(tage=a,peraa=True)
    
flam = F_lam_per_M(fl, wv * (1 + specz), specz, 0, sp.stellar_mass)*10**logmass
    
###########gen spec##########
Gs = Gen_SF_spec(field, galaxy, 1, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
        phot_errterm = 0.04, irac_err = 0.08) 

Gs.Make_sim(wv,flam,specz,perturb = True , rndstate=np.random.randint(0,10000,1)[0])

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_simgrism_data(Gs)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 19, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 19, nlive_points = 4000,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=8), queue_size=8)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_SFfit_sim_p1'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFfit_sim_p1_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, blwa = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFfit_sim_p1_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, blwa, dres.logl[-1]])
