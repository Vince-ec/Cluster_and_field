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
#########BALMER fit slope########
#################################

start = time()
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    lines = sys.argv[4:]
    
beams = mfit_path + '{}_{}.beams.fits'.format(field, galaxy)

#############multifit###############
mb_g102, mb_g141 = Gen_multibeams(beams, args = args)

wave0 = 4000
Q_temps = Gen_temp_dict_balm(specz,8000,16000, lines = lines)
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
   
    sb = Gaussian_prior(u[14], [-0.2, 0.2], 0, 0.025)
    sr = Gaussian_prior(u[15], [-0.2, 0.2], 0, 0.025)

    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr = X
    
    wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])
    
    Q_temps['fsps_model'] = SpectrumTemplate(wave=wave, flux=flux + sb*flux*(wave-wave0)/wave0)
    
    g102_fit = mb_g102.template_at_z(z, templates = Q_temps, fitter='lstsq')
    
    Q_temps['fsps_model'] = SpectrumTemplate(wave=wave, flux=flux + sr*flux*(wave-wave0)/wave0)
    
    g141_fit = mb_g141.template_at_z(z, templates = Q_temps, fitter='lstsq')

    Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flux)

    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)

    return  -(g102_fit['chi2'] + g141_fit['chi2'] + np.sum((((Gs.Pflx - Pmfl*scl) / Gs.Perr)**2))) / 2
   
#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 16, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi', pool=Pool(processes=12), queue_size=12)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 14,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=12), queue_size=12)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_tabMfit'.format(field, galaxy), dres) 

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10','z', 'd', 'sb', 'sr']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_tabMfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd, bfsb, bfsr= dres.samples[-1]

np.save(pos_path + '{0}_{1}_tabMfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bfz, bfd, bfsb, bfsr dres.logl[-1]])

#### gen light-weighted age posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

lwa = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr = dres.samples[i]
    lwa.append(get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a),sp)[0])


t,pt = Get_lwa_posterior(np.array(lwa), dres)
np.save(pos_path + '{0}_{1}_tabMfit_Plwa'.format(field, galaxy),[t,pt])

#### gen logmass posterior
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
lm = []
for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr = dres.samples[i]

    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    flam = F_lam_per_M(flux, wave * (1+z), z, 0, sp.stellar_mass)
    Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flam)
    scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
    lm.append(np.log10(scl))

t,pt = Get_lwa_posterior(np.array(lm), dres)
np.save(pos_path + '{0}_{1}_tabMfit_Plm'.format(field, galaxy),[t,pt])

end = time()
print(end - start)