#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
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

#############multifit###############
args = np.load('/home/vestrada78840/ce_scripts/fit_args.npy')[0]
mb = multifit.MultiBeam('/home/vestrada78840/ce_scripts/gdn-grism-j123656p6215_14138.beams.fits',**args)

grism_beams = {}
for g in mb.PA:
    grism_beams[g.lower()] = []
    for pa in mb.PA[g]:
        for i in mb.PA[g][pa]:
            grism_beams[g.lower()].append(mb.beams[i])

mb_g102 = multifit.MultiBeam(grism_beams['g102'], fcontam=mb.fcontam, 
                             min_sens=mb.min_sens, min_mask=mb.min_mask, 
                             group_name=mb.group_name+'-g102')
# bug, will be fixed ~today to not have to do this in the future
for b in mb_g102.beams:
    if hasattr(b, 'xp'):
        delattr(b, 'xp')
mb_g102.initialize_masked_arrays()

mb_g141 = multifit.MultiBeam(grism_beams['g141'], fcontam=mb.fcontam, 
                             min_sens=mb.min_sens, min_mask=mb.min_mask, 
                             group_name=mb.group_name+'-g141')
# bug, will be fixed ~today to not have to do this in the future
for b in mb_g141.beams:
    if hasattr(b, 'xp'):
        delattr(b, 'xp')
mb_g141.initialize_masked_arrays()
wave0 = 4000

tilt_temps = {}

entries = ['line SII','line Ha','line OI-6302','line HeI-5877',
           'line OIII','line Hb','line OIII-4363','line Hg','line Hd','line NeIII-3867','line OII']

for k in args['t1']:
    if k in entries:
        tilt_temps[k] = args['t1'][k]

def spec_construct(g102_fit,g141_fit, z, wave0 = 4000, ):
    flat = np.ones_like(g141_fit['cont1d'].wave)
    slope = flat*(g141_fit['cont1d'].wave/(1+z)-wave0)/wave0
    tilt = flat * g141_fit['cfit']['fsps_model'][0]+slope * g141_fit['cfit']['fsps_model_slope'][0]
    untilted_continuum = g141_fit['cont1d'].flux / tilt

    line_g141 = (g141_fit['line1d'].flux - g141_fit['cont1d'].flux)/g141_fit['cont1d'].flux
    untilted_line_g141 = untilted_continuum*(1+line_g141)


    flat = np.ones_like(g102_fit['cont1d'].wave)
    slope = flat*(g102_fit['cont1d'].wave/(1+z)-wave0)/wave0
    tilt = flat * g102_fit['cfit']['fsps_model'][0]+slope * g102_fit['cfit']['fsps_model_slope'][0]
    untilted_continuum = g102_fit['cont1d'].flux / tilt

    line_g102 = (g102_fit['line1d'].flux - g102_fit['cont1d'].flux)/g102_fit['cont1d'].flux
    untilted_line_g102 = untilted_continuum*(1+line_g102)

    FL = np.append(untilted_line_g102[g102_fit['cont1d'].wave <= 12000],untilted_line_g141[g102_fit['cont1d'].wave > 12000])
    return g102_fit['cont1d'].wave, FL

############build priors#############
def Z_prior_mu(lmass):
    M = [9, 11.5]
    P = [-0.5813, 0.06818]
    iP = interp1d(M,P)
    return iP(lmass) 

onesig = (0.04 + 0.47)/2
mllim = np.log10(0.001 / 0.019)
mhlim = np.log10(0.031 / 0.019)

zscale = 0.01
agelim = Oldest_galaxy(specz)

def Galfit_prior(u):
    m = 10**Gaussian_prior(u[0], [mllim, mhlim], Z_prior_mu(logmass), onesig)
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))

    d = 4*u[8]
    z = stats.norm.ppf(u[9],loc = specz, scale = zscale)
    
    #ba = log_10_prior(u[10], [0.1,10])
    #ra = log_10_prior(u[11], [0.1,10])
    
    return [m, a, m1, m2, m3, m4, m5, m6, d, z]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, d, z = X
    
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    tilt_temps['fsps_model_slope'] = SpectrumTemplate(wave=wave, flux=flux*(wave-wave0)/wave0)
    tilt_temps['fsps_model'] = SpectrumTemplate(wave, flux)
    
    g102_fit = mb_g102.template_at_z(z, templates=tilt_temps, fitter='lstsq')
    g141_fit = mb_g141.template_at_z(z, templates=tilt_temps, fitter='lstsq')

    wv_obs, flx = spec_construct(g102_fit,g141_fit,z)

    Pmfl = Gs.Sim_phot_mult(wv_obs, flx)

    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)

    return  -(g102_fit['chi2'] + g141_fit['chi2'] + np.sum((((Gs.Pflx - Pmfl*scl) / Gs.Perr)**2))) / 2

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0

###########gen spec##########
Gs = Gen_SF_spec(field, galaxy, 1, g102_lims=[8200, 11300], g141_lims=[11200, 16000]) 

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 10, nlive_points = 4000,
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

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'd', 'z']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFMfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFMfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfd, bfz, dres.logl[-1]])
