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
    SFa = str(sys.argv[4])
    dust = float(sys.argv[5])
    metal = float(sys.argv[6])

verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
zscale = 0.1 * (1 + specz)

lbt, sfh = np.load(data_path + 'sim_SFH/SFH_z{}_{}.npy'.format(specz, galaxy),allow_pickle = True)
#lbt, sfh = np.load('../sim_SFH/SFH_z{}_{}.npy'.format(specz, galaxy),allow_pickle = True)

sfh = sfh[1:]
lbt = lbt[:-1]

if SFa == 'Q':
    def Galfit_prior(u):
        m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
        a = (agelim - 1)* u[1] + 1

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

        Pmfl = Gs.Sim_phot_mult(wave * (1 + z), F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm)

        return lnlike_phot(Gs.Pflx, Gs.Perr, Pmfl)

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
        a = (agelim - 1)* u[1] + 1

        tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
        taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
        m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))

        lm = Gaussian_prior(u[8], [8.0, 12.5], 11, 0.75)
        z = stats.norm.ppf(u[9],loc = specz, scale = zscale)

        d = 4*u[10]

        return [m, a, m1, m2, m3, m4, m5, m6, lm, z, d]

    def Galfit_L(X):
        m, a, m1, m2, m3, m4, m5, m6, lm, z, d= X

        sp.params['dust2'] = d
        sp.params['logzsol'] = np.log10(m)

        time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

        sp.set_tabular_sfh(time,sfr) 

        wave, flux = sp.get_spectrum(tage = a, peraa = True)
        Pmfl = Gs.Sim_phot_mult(wave * (1 + z), F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm)

        return lnlike_phot(Gs.Pflx, Gs.Perr, Pmfl)


############ simulate data with no offsets#############
def spec_sim_werr(Gs,lbt,sfh, dust, metal, SFa, lm = 11, z = 1.5):
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)
    sp.set_tabular_sfh(lbt, sfh[::-1])
    sp.params['dust2'] = dust
    
    if SFa == 'Q':
        sp.params['dust1'] = dust
    else:
        sp.params['dust1'] = 0
        
    sp.params['logzsol'] = np.log10(metal)

    wave, flux = sp.get_spectrum(tage = lbt[-1], peraa = True)
    flam = F_lam_per_M(flux,wave*(1+z),z,0,10**lm)*10**lm

    Smfl, Pmfl = Full_forward_model(Gs, wave,flam, z, 
                                    wvs, flxs, errs, beams, trans)
    
    Sc =  Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
    
    for i in range(len(Smfl)):
        Smfl[i] = Smfl[i] + np.random.normal(0,errs[i])/Sc
        Pmfl[i] = Pmfl[i] + np.random.normal(0,Gs.Perr[i])/Sc
    
    return wvs, Smfl, errs/Sc, beams, trans, Pmfl, Gs.Perr/Sc

#########define fsps and gen spec#########
if SFa == 'Q':
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    Gs = Gen_spec_2D('GND', 21156, specz, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                 phot_errterm = 0.04, irac_err = 0.08, mask = False)
else:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    if specz == 1.0:
        #maybe 16041
        Gs = Gen_spec_2D('GND',37006, 1.0, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
             phot_errterm = 0.04, irac_err = 0.08, mask = True)
    if specz == 1.5:
        Gs = Gen_spec_2D('GND',27930, 1.5, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                     phot_errterm = 0.04, irac_err = 0.08, mask = True)
    if specz == 2.0:
        Gs = Gen_spec_2D('GND', 19591, 2.0, g102_lims=[8200, 11300], g141_lims=[11200, 16000],
             phot_errterm = 0.04, irac_err = 0.08, mask = True)

wvs, flxs, errs, beams, trans, Spmfl, Spmerr = spec_sim_werr(Gs, lbt, sfh, dust, metal, SFa, lm = logmass, z=specz)
Gs.Pflx = Spmfl
Gs.Perr = Spmerr

#######set up dynesty########

if SFa == 'Q':
    sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 15, nlive_points = 4000,
                                             sample = 'rwalk', bound = 'multi',
                                             pool=Pool(processes=8), queue_size=8)
else:
    sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 11, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results

np.save(out_path + 'z{}_{}_Ifit_imp_phot'.format(specz, galaxy), dres) 

##save out P(z) and bestfit##

if SFa == 'Q':
    fit_dict = {}
    params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm', 'z', 'd']
    P_params = ['Pm', 'Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Pm7', 'Pm8', 'Pm9', 'Pm10', 'Plm', 'Pz', 'Pd']
    bf_params = ['bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 'bfm7', 'bfm8', 'bfm9', 'bfm10', 'bflm', 'bfz', 'bfd']

else:
    fit_dict = {}
    params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'z', 'd']
    P_params = ['Pm', 'Pa', 'Pm1', 'Pm2', 'Pm3', 'Pm4', 'Pm5', 'Pm6', 'Plm', 'Pz', 'Pd']
    bf_params = ['bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 'bflm', 'bfz', 'bfd']
    
bfits = dres.samples[-1]

for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    fit_dict[params[i]] = t
    fit_dict[P_params[i]] = pt
    fit_dict[bf_params[i]] = bfits[i]

np.save(pos_path + 'z{}_{}_Ifit_impfits_phot'.format(specz, galaxy),fit_dict)
