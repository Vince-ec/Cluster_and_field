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
    galaxy = int(sys.argv[1])
    specz = float(sys.argv[2])
    logmass = float(sys.argv[3])
    lwa = float(sys.argv[4])
    
verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
zscale = 0.0035 * (1 + specz)

lbt, sfh = np.load(data_path + 'sim_SFH/SFH_{}.npy'.format(galaxy),allow_pickle = True)
#lbt, sfh = np.load('../sim_SFH/SFH_{}.npy'.format(galaxy),allow_pickle = True)

sfh = sfh[1:]
lbt = lbt[:-1]

if lwa > 1:
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
        sfh, timeax = tuple_to_sfh_stand_alone(sfh_tuple, z)

        sp.set_tabular_sfh(timeax,sfh) 

        wave, flux = sp.get_spectrum(tage = timeax[-1], peraa = True)

        pmfl = Gs.Sim_phot_mult(wave * (1 + z),flux)

        SC = Scale_model(Gs.Pflx, Gs.Perr, pmfl)

        Gmfl, Pmfl = Full_forward_model(Gs, wave, flux*SC, z, 
                                        wvs, flxs, errs, beams, trans)

        Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)

        return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

else:
    def Z_prior_mu(lmass):
        M = [9, 11.5]
        P = [-0.5813, 0.06818]
        iP = interp1d(M,P)
        return iP(lmass) 

    onesig = (0.04 + 0.47)/2
    mllim = np.log10(0.001 / 0.019)
    mhlim = np.log10(0.031 / 0.019)
    
    def Galfit_prior(u):
        m = 10**Gaussian_prior(u[0], [mllim, mhlim], Z_prior_mu(logmass), onesig)

        t25, t50, t75 = get_tx_vals([u[1],u[2],u[3]])

        logssfr = -15 + 8*u[4]

        d = log_10_prior(u[5],[1E-3,2])

        bp1 = Gaussian_prior(u[6], [-0.1,0.1], 0, 0.05)
        rp1 = Gaussian_prior(u[7], [-0.05,0.05], 0, 0.025)

        ba = log_10_prior(u[8], [0.1,10])
        bb = log_10_prior(u[9], [0.0001,1])
        bl = log_10_prior(u[10], [0.01,1])

        ra = log_10_prior(u[11], [0.1,10])
        rb = log_10_prior(u[12], [0.0001,1])
        rl = log_10_prior(u[13], [0.01,1])

        return [m, t25, t50, t75, logssfr, d, bp1, rp1, ba, bb, bl, ra, rb, rl]

    def Galfit_L(X):
        m, t25, t50, t75, logssfr, d, bp1, rp1, ba, bb, bl, ra, rb, rl = X

        sp.params['dust2'] = d
        sp.params['logzsol'] = np.log10(m)

        sfh_tuple = np.hstack([0, logssfr, 3, t25,t50,t75])
        sfh, timeax = tuple_to_sfh_stand_alone(sfh_tuple, 1.5)

        sp.set_tabular_sfh(timeax,sfh) 

        wave, flux = sp.get_spectrum(tage = timeax[-1], peraa = True)

        pmfl = Gs.Sim_phot_mult(wave * (1 + 1.5),flux)

        SC = Scale_model(Gs.Pflx, Gs.Perr, pmfl)

        Gmfl, Pmfl = Full_forward_model(Gs, wave, flux*SC, 1.5, 
                                        wvs, flxs, errs, beams, trans)

        Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)

        return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)


############ simulate data with no offsets#############
def spec_sim_werr(Gs,lbt,sfh, dust, metal, lwa, lm = 11, z = 1.5):
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)
    sp.set_tabular_sfh(lbt, sfh[::-1])
    sp.params['dust2'] = dust
    
    if lwa > 0:
        sp.params['dust1'] = dust
    else:
        sp.params['dust1'] = 0
        
    sp.params['logzsol'] = np.log10(metal)

    wave, flux = sp.get_spectrum(tage = lbt[-1], peraa = True)
    flam = F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm

    Smfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
    
    Sc =  Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
    print(Sc)
    
    for i in range(len(Smfl)):
        Smfl[i] = Smfl[i] + np.random.normal(0,errs[i])/Sc
        
    Pmfl = Pmfl + np.random.normal(0,Gs.Perr)/Sc
    
    return wvs, Smfl, errs/Sc, beams, trans, Pmfl, Gs.Perr/Sc
#########define fsps and gen spec#########
if lwa > 1:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    Gs = Gen_spec_2D('GSD', 39170, 1.5, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                 phot_errterm = 0.04, irac_err = 0.08, mask = False)
else:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    Gs = Gen_spec_2D('GND',27930, 1.5, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                     phot_errterm = 0.04, irac_err = 0.08, mask = True)

wvs, flxs, errs, beams, trans, Spmfl, Spmerr = spec_sim_werr(Gs, lbt,sfh,0.1,1, lwa)
Gs.Pflx = Spmfl
Gs.Perr = Spmerr

#######set up dynesty########

if lwa > 1:
    print('Q')
    sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 15, nlive_points = 4000,
                                             sample = 'rwalk', bound = 'multi',
                                             pool=Pool(processes=8), queue_size=8)
else:
    print('SF')
    sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 14, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)
    
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=False)

dres = sampler.results

np.save(out_path + '{}_Ifit_impKI'.format(galaxy), dres) 

