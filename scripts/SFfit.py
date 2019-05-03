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

def Galfit_prior(u):
    m = (0.03*u[0] +0.001 )/ 0.019
    a = (3)* u[1] + 0.5

    #t = Gaussian_prior(u[2], [0.01,3],1,.5)
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))
    
    lm = Gaussian_prior(u[8], [8.0, 12.5], 11, 0.75)
  
    #z = stats.norm.ppf(u[4],loc = specz, scale = zscale)
    
    d = 4*u[9]
    
    bp1 = Gaussian_prior(u[10], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[11], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[12], [0.1,10])
    bb = log_10_prior(u[13], [0.0001,1])
    bl = log_10_prior(u[14], [0.01,1])
    
    ra = log_10_prior(u[15], [0.1,10])
    rb = log_10_prior(u[16], [0.0001,1])
    rl = log_10_prior(u[17], [0.01,1])
   
    #lwa = get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a),sp)[0]
    
    return [m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl = X
    
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a,binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm, specz, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

###########gen spec##########
Gs = Gen_SF_spec(field, galaxy, 1, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
        phot_errterm = 0.04, irac_err = 0.08) 

####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 18, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 13, nlive_points = 4000,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=8), queue_size=8)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_SFfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, dres.logl[-1]])