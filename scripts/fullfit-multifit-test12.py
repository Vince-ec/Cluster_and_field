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

#################################
############g141###############
#################################

start = time()
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    
if hpath == '/home/vestrada78840/':
    beams = '/home/vestrada78840/ce_scripts/gdn-grism-j123656p6215_25319.beams.fits'
else:
    beams = '../data/multifit_data/gdn-grism-j123656p6215_25319.beams.fits'
#############multifit###############
mb_g102, mb_g141 = Gen_multibeams(beams, args = args)

wave0 = 4000
Q_temps = {}

####################################
agelim = Oldest_galaxy(specz)
zscale = 0.005 

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8], u[9], u[10],u[11]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a))
  
    z = Gaussian_prior(u[12], [specz - 0.01, specz + 0.01], specz, zscale)
    
    d = log_10_prior(u[13],[1E-3,2])

    #ba = log_10_prior(u[14], [0.1,10])
    #ra = log_10_prior(u[15], [0.1,10])
   
    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, = X
    
    wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])
    
    Q_temps['fsps_model'] = SpectrumTemplate(wave, flux)
    Q_temps['fsps_model_slope'] = SpectrumTemplate(wave=wave, flux=flux*(wave-wave0)/wave0)
    
    g141_fit = mb_g141.template_at_z(z, templates = Q_temps, fitter='lstsq')

    return  -(g141_fit['chi2']) / 2
   
#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
#Gs = Gen_spec(field, galaxy, 1) 

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 14, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',pool=Pool(processes=12), queue_size=12)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 14,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=12), queue_size=12)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_tabMfit12'.format(field, galaxy), dres) 

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10','z', 'd']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_tabMfit12_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd= dres.samples[-1]

np.save(pos_path + '{0}_{1}_tabMfit12_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd, dres.logl[-1]])
end = time()
print(end - start)