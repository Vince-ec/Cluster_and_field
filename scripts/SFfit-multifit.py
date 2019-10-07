#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from grizli import multifit
from grizli.utils import SpectrumTemplate
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    logmass = float(sys.argv[4])
    
verbose=False
poolsize = 8

#############
###########gen spec##########
Gs = Gen_spec_2D(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 
    
MBS = Gather_MB_data(Gs)   
#############multifit#######

if Gs.g102 and Gs.g141:
    def spec_construct(fits, z, slopes, wave0 = 4000, usetilt = True):
        fluxes = []

        for i in range(len(fits)):
            flat = np.ones_like(fits[i]['cont1d'].wave) # ones array
            slope = (fits[i]['cont1d'].wave/(1+z)-wave0)/wave0 # slope without coeff 
            tilt = fits[i]['cfit']['fsps_model'][0]*(flat + (slope * slopes[i])) # scaling up and slope coeff
            untilted_continuum = fits[i]['cont1d'].flux / tilt # return to fsps scale
            lines = (fits[i]['line1d'].flux - fits[i]['cont1d'].flux)/fits[i]['cont1d'].flux
            fluxes.append(untilted_continuum*(1+lines))

        FL = np.append(fluxes[0][fits[0]['cont1d'].wave <= 12000],fluxes[1][fits[0]['cont1d'].wave > 12000])
        return fits[0]['cont1d'].wave, FL
    
if Gs.g102 and not Gs.g141:
    def spec_construct(fits, z, slopes, wave0 = 4000, usetilt = True):
        fluxes = []

        for i in range(len(fits)):
            flat = np.ones_like(fits[i]['cont1d'].wave) # ones array
            slope = (fits[i]['cont1d'].wave/(1+z)-wave0)/wave0 # slope without coeff 
            tilt = fits[i]['cfit']['fsps_model'][0]*(flat + (slope * slopes[i])) # scaling up and slope coeff
            untilted_continuum = fits[i]['cont1d'].flux / tilt # return to fsps scale
            lines = (fits[i]['line1d'].flux - fits[i]['cont1d'].flux)/fits[i]['cont1d'].flux
            fluxes.append(untilted_continuum*(1+lines))

        FL = fluxes[0]
        return fits[0]['cont1d'].wave, FL
    
if Gs.g141 and not Gs.g102:
    def spec_construct(fits, z, slopes, wave0 = 4000, usetilt = True):
        fluxes = []

        for i in range(len(fits)):
            flat = np.ones_like(fits[i]['cont1d'].wave) # ones array
            slope = (fits[i]['cont1d'].wave/(1+z)-wave0)/wave0 # slope without coeff 
            tilt = fits[i]['cfit']['fsps_model'][0]*(flat + (slope * slopes[i])) # scaling up and slope coeff
            untilted_continuum = fits[i]['cont1d'].flux / tilt # return to fsps scale
            lines = (fits[i]['line1d'].flux - fits[i]['cont1d'].flux)/fits[i]['cont1d'].flux
            fluxes.append(untilted_continuum*(1+lines))

        FL = fluxes[0]
        return fits[0]['cont1d'].wave, FL

##############
#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

########
wave0 = 4000
SF_temps =  Gen_temp_dict(specz, 8000, 16000)
############build priors#############
def Z_prior_mu(lmass):
    M = [9, 11.5]
    P = [-0.5813, 0.06818]
    iP = interp1d(M,P)
    return iP(lmass) 

onesig = (0.04 + 0.47)/2
mllim = np.log10(0.001 / 0.019)
mhlim = np.log10(0.031 / 0.019)

zscale = 0.005 
agelim = Oldest_galaxy(specz)

def Galfit_prior(u):
    m = 10**Gaussian_prior(u[0], [mllim, mhlim], Z_prior_mu(logmass), onesig)
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))

    d = 4*u[8]

    z = stats.norm.ppf(q = u[9], loc = specz, scale = zscale)

    s1 = Gaussian_prior(u[10], [-0.2, 0.2], 0, 0.025)
    s2 = Gaussian_prior(u[11], [-0.2, 0.2], 0, 0.025)
    
    return [m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = X
    
    wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6], agebins = 6, SF = True)
    
    Gchi2, fits = Fit_MB(MBS, [s1, s2], SF_temps, wave, flux, z)

    wv_obs, flx = spec_construct(fits, z, [s1, s2])

    Pmfl = Gs.Sim_phot_mult(wv_obs, flx)

    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)

    return  -(Gchi2+ np.sum((((Gs.Pflx - Pmfl*scl) / Gs.Perr)**2))) / 2

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 12, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=12), queue_size=12)
                                      
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 10,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=12), queue_size=12)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_SFMfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'd', 'z', 's1', 's2']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFMfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz, bfs1, bfs2 = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFMfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz, bfs1, bfs2, dres.logl[-1]])

#### gen light-weighted age posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
lwa = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]
    lwa.append(get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum = 6),sp)[0])

t,pt = Get_lwa_posterior(np.array(lwa), dres)
np.save(pos_path + '{0}_{1}_SFMfit_Plwa'.format(field, galaxy),[t,pt])

#### gen logmass posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
lm = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]

    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    flam = F_lam_per_M(flux, wave * (1+z), z, 0, sp.stellar_mass)
    Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flam)
    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
    lm.append(np.log10(scl))

t,pt = Get_lwa_posterior(np.array(lm), dres)
np.save(pos_path + '{0}_{1}_SFMfit_Plm'.format(field, galaxy),[t,pt])

#### gen logmass posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

lines = []
for i in range(len(MB)):
    lines.append({})    
    for k in SF_temps:
        if k[0] == 'l':
            lines[i][k] = []

for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]
    wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6], agebins = 6, SF = True)
      
    Gchi2, fits = Fit_MB(MBS, [s1, s2], SF_temps, wave, flux, z)

    for ii in range(len(lines)):
        for k in lines[ii]:
            lines[ii][k].append(fit[ii]['cfit'][k][0])

for i in range(len(lines)):
    for k in lines[i]:
        if sum(lines[i][k]) > 0:
            x,px = Get_derived_posterior(np.array(lines[i][k]), dres)
            np.save(pos_path + '{}_{}_SFMfit_P{}_{}'.format(field, galaxy, k, i),[x,px])

end = time()
print(end - start)