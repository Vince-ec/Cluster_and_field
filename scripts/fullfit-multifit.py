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
    
verbose=False

#############multifit###############
args = np.load('/home/vestrada78840/ce_scripts/fit_args.npy')[0]
mb = multifit.MultiBeam('/home/vestrada78840/ce_scripts/gdn-grism-j123656p6215_25319.beams.fits',**args)

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

tilt_temps = {}
wave0 = 4000
####################################
agelim = Oldest_galaxy(specz)
zscale = 0.035 * (1 + specz)

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8], u[9], u[10],u[11]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a))
  
    z = stats.norm.ppf(u[12],loc = specz, scale = zscale)
    
    d = log_10_prior(u[13],[1E-3,2])

    ba = log_10_prior(u[14], [0.1,10])
    ra = log_10_prior(u[15], [0.1,10])
   
    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, ba, ra]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, ba, ra = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    tilt_temps['fsps_model_slope'] = SpectrumTemplate(wave=wave, flux=flux*(wave-wave0)/wave0)
    tilt_temps['fsps_model'] = SpectrumTemplate(wave, flux)
    
    g102_fit = mb_g102.template_at_z(z, templates=tilt_temps, fitter='lstsq')
    g141_fit = mb_g141.template_at_z(z, templates=tilt_temps, fitter='lstsq')

    Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flux)

    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)

    return  -(g102_fit['chi2']*(1/ba) + g141_fit['chi2']*(1/ra) + np.sum((((Gs.Pflx - Pmfl*scl) / Gs.Perr)**2))) / 2
   
#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 16, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=12), queue_size=12)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 14,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=12), queue_size=12)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_tabMfit'.format(field, galaxy), dres) 

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10','z', 'd', 'ba', 'ra']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_tabMfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd, bfba, bfra= dres.samples[-1]

np.save(pos_path + '{0}_{1}_tabMfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd, bfba, bfra, dres.logl[-1]])
