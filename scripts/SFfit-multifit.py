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
def spec_construct(g102_fit,g141_fit, z, sb, sr, wave0 = 4000, usetilt = True):
    flat = np.ones_like(g141_fit['cont1d'].wave) # ones array
    slope = (g141_fit['cont1d'].wave/(1+z)-wave0)/wave0 # slope without coeff 
    tilt = g141_fit['cfit']['fsps_model'][0]*(flat + (slope * sr)) # scaling up and slope coeff
    untilted_continuum = g141_fit['cont1d'].flux / tilt # return to fsps scale
    line_g141 = (g141_fit['line1d'].flux - g141_fit['cont1d'].flux)/g141_fit['cont1d'].flux
    untilted_line_g141 = untilted_continuum*(1+line_g141)


    flat = np.ones_like(g102_fit['cont1d'].wave)
    slope = (g102_fit['cont1d'].wave/(1+z)-wave0)/wave0
    tilt = g102_fit['cfit']['fsps_model'][0]*(flat + slope * sb)
    untilted_continuum = g102_fit['cont1d'].flux / tilt

    line_g102 = (g102_fit['line1d'].flux - g102_fit['cont1d'].flux)/g102_fit['cont1d'].flux
    untilted_line_g102 = untilted_continuum*(1+line_g102)

    FL = np.append(untilted_line_g102[g102_fit['cont1d'].wave <= 12000],untilted_line_g141[g102_fit['cont1d'].wave > 12000])

    return g102_fit['cont1d'].wave, FL
##############
#############multifit###############
beams = mfit_path + '{}_{}.beams.fits'.format(field, galaxy)

mb_g102, mb_g141 = Gen_multibeams(beams, args = args)

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
    z = Gaussian_prior(u[9], [specz - 0.01, specz + 0.01], specz, zscale)
    
    sb = Gaussian_prior(u[10], [-0.2, 0.2], 0, 0.025)
    sr = Gaussian_prior(u[11], [-0.2, 0.2], 0, 0.025)
    
    return [m, a, m1, m2, m3, m4, m5, m6, d, z, sb, sr]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, d, z, sb, sr = X
    
    wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6], agebins = 6, SF = True)
    
    SF_temps['fsps_model'] = SpectrumTemplate(wave, flux + sb*flux*(wave-wave0)/wave0)    
    g102_fit = mb_g102.template_at_z(z, templates = SF_temps, fitter='lstsq')
    
    SF_temps['fsps_model'] = SpectrumTemplate(wave, flux + sr*flux*(wave-wave0)/wave0)    
    g141_fit = mb_g141.template_at_z(z, templates = SF_temps, fitter='lstsq')

    wv_obs, flx = spec_construct(g102_fit,g141_fit,z, sb, sr)

    Pmfl = Gs.Sim_phot_mult(wv_obs, flx)

    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)

    return  -(g102_fit['chi2'] + g141_fit['chi2'] + np.sum((((Gs.Pflx - Pmfl*scl) / Gs.Perr)**2))) / 2

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

###########gen spec##########
Gs = Gen_SF_spec(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 

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

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'd', 'z', 'sb', 'sr']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFMfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz, bfsb, bfsr = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFMfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz, bfsb, bfsr, dres.logl[-1]])

#### gen light-weighted age posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
lwa = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, d, z, sb, sr = dres.samples[i]
    lwa.append(get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum = 6),sp)[0])

t,pt = Get_lwa_posterior(np.array(lwa), dres)
np.save(pos_path + '{0}_{1}_SFMfit_Plwa'.format(field, galaxy),[t,pt])

#### gen logmass posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
lm = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, d, z, sb, sr = dres.samples[i]

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

end = time()
print(end - start)