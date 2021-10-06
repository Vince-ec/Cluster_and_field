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
START = time()
    
if __name__ == '__main__':
    specz = float(sys.argv[1])
    form = sys.argv[2]
    
verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
zscale = 0.035 * (1 + specz)

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (agelim)*(1 - u[1]*0.15)

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8], u[9], u[10],u[11]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a))

    lm = Gaussian_prior(u[12], [9.5, 12.5], 11, 0.75)

    z = stats.norm.ppf(u[13],loc = specz, scale = zscale)

    d = log_10_prior(u[14],[1E-3,2])

    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d = X

    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    #need to fix        

    flam = F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm
    
    Pwv, Pmfl = GS.forward_model_phot(wave*(1+z), flam)
    Rmfl = forward_model_all_beams(GS.RBEAMS, GS.RTRANS, GS.Rwv, wave*(1+z), flam)

    return lnlike_phot(GS.Pflx, GS.Per, Pmfl) + lnlike_phot(GS.Rflx, GS.Rer, Rmfl) 


#########define fsps and gen spec#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 4, dust_type = 2)
if form == 'early':
    sp.params['tau'] = 0.1
    
if form == 'late':
    sp.params['tau'] = 0.5

sp.params['dust2'] = 0.3
sp.params['dust1'] = 0.3
wave, flux = sp.get_spectrum(tage = agelim, peraa=True)
flam = F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**11

sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
GS = Gen_spec_NIRISS_sim(specz, wave, flam, Iused = ['F200W'], Fused = ['F200W', 'F444W'], I_lims=[[17700, 22100]], random_seed = 3)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 15, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results

np.save(out_path + 'NIRISS_sim_st2_z{}_{}'.format(specz, form), dres) 

# ##save out P(z) and bestfit##

fit_dict = {}
params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm', 'z', 'd']
P_params = ['Pm', 'Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm', 'Pz', 'Pd']
bf_params = ['bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 'bfm7', 'bfm8', 'bfm9', 'bfm10', 'bflm', 'bfz', 'bfd']

bfits = dres.samples[-1]

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    fit_dict[params[i]] = t
    fit_dict[P_params[i]] = pt
    fit_dict[bf_params[i]] = bfits[i]

np.save(pos_path + 'NIRISS_sim_st2_z{}_{}'.format(specz, form),fit_dict)
