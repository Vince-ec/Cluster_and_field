#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_exam import Gen_ALMA_spec
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
import fsps
import dynesty
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy import stats
from sim_engine import forward_model_grism, Salmon
from spec_id import *
from spec_tools import Oldest_galaxy
from astropy.cosmology import Planck13 as cosmo
from multiprocessing import Pool
from prospect.models.transforms import logsfr_ratios_to_masses
from spec_stats import Get_posterior

hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/fdata/scratch/vestrada78840/chidat/'
    pos_path = '/home/vestrada78840/posteriors/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/out_dict/'
    pos_path = '../data/posteriors/'
    phot_path = '../phot/'

if __name__ == '__main__':
    galaxy_id = int(sys.argv[1])
    lim1 = int(sys.argv[2])
    lim2 = int(sys.argv[3])

Gs = Gen_ALMA_spec(galaxy_id, 1, g102_lims=[8750,11300], g141_lims=[lim1,lim2], mdl_err=False)

sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1),sfh = 0, dust_type = 1)

############
###priors###
def alma_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (3 - 1)* u[1] + 1
    bsc= Gaussian_prior(u[2], [0.8, 1.2], 1, 0.05)
    rsc= Gaussian_prior(u[3], [0.8, 1.2], 1, 0.05)
    bp1 = Gaussian_prior(u[4], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[5], [-0.05,0.05], 0, 0.025)
    
    lm = Gaussian_prior(u[6], [9.5, 12.5], 11, 0.75)
       
    z = stats.norm.ppf(u[7],loc = 1.6, scale = 0.1)
    
    d = log_10_prior(u[8],[1E-3,2])
    
    return [m, a, bsc, rsc, bp1, rp1, lm, z, d]

############
#likelihood#
def Full_forward_model(spec, wave, flux, specz):
    Gmfl = []
    
    for i in range(len(wvs)):
        Gmfl.append(forward_model_all_beams(beams[i], trans[i], wvs[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl

def Full_calibrate(Gmfl, p1):
    for i in range(len(wvs)):
        Gmfl[i] = Gmfl[i] * ((p1[i] * wvs[i]) / (wvs[i][-1] - wvs[i][0]) + 5)
    return Gmfl

def Full_calibrate_2(Gmfl, p1, sc):
    for i in range(len(wvs)):
        rGmfl= Gmfl[i] * (p1[i] * (wvs[i] -(wvs[i][-1] + wvs[i][0])/2 ) + 1E3)
        scale = Scale_model(Gmfl[i],np.ones_like(Gmfl[i]),rGmfl)
        Gmfl[i] = scale * rGmfl * sc[i]
    return Gmfl

def Calibrate_grism(spec, Gmfl, p1):
    linecal = []
    for i in range(len(wvs)):
        lines = ((p1[i] * wvs[i]) / (wvs[i][-1] - wvs[i][0]) + 5)
        scale = Scale_model(flxs[i]  / lines, errs[i] / lines, Gmfl[i])    
        linecal.append(scale * lines)
        
    return linecal


def Full_fit(spec, Gmfl, Pmfl):
    Gchi = 0
    
    for i in range(len(wvs)):
        scale = Scale_model(flxs[i], errs[i], Gmfl[i])
        Gchi = Gchi + np.sum(((((flxs[i] / scale) - Gmfl[i]) / (errs[i] / scale))**2))
    
    Pchi = np.sum((((spec.Pflx - Pmfl) / spec.Perr)**2))

    return Gchi, Pchi

def Full_fit_2(spec, Gmfl, Pmfl, a, b, l): 
    Gln = 0
    
    for i in range(len(wvs)):
        scale = Scale_model(flxs[i], errs[i], Gmfl[i])
        noise = noise_model(np.array([wvs[i],flxs[i], errs[i]]).T, Gmfl[i] * scale)
        noise.GP_exp_squared(a[i],b[i],l[i])
        Gln += noise.gp.lnlikelihood(noise.diff)

    Pln = lnlike_phot(spec.Pflx, spec.Perr, Pmfl)
    
    return Gln + Pln

def Full_fit_3(spec, Gmfl, Pmfl):
    Gchi = 0
    
    for i in range(len(wvs)):
        Gchi = Gchi + np.sum( ((flxs[i] - Gmfl[i]) / errs[i] )**2 )
    
    Pchi = np.sum( ((spec.Pflx - Pmfl) / spec.Perr)**2)

    return Gchi, Pchi

wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)
print(wvs)
def alma_L(X):
    m, a, bsc, rsc, bp1, rp1, lm, z, d = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux, wave*(1+z) , z, 0, sp.stellar_mass)*10**lm, z)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], [bsc, rsc])

    Gchi, Pchi = Full_fit_3(Gs, Gmfl, Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

############
####run#####
sampler = dynesty.DynamicNestedSampler(alma_L, alma_prior, ndim = 9, sample = 'rwalk', bound = 'single',
                                  queue_size = 8, pool = Pool(processes=8))  
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results
############
####save####
np.save(out_path + 'ALMA_{0}'.format(galaxy_id), dres) 

############# 
#get lightweighted age
#############

params = ['m', 'a', 'bsc', 'rsc', 'bp1', 'rp1', 'lm', 'z', 'd']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + 'ALMA_{0}_P{1}'.format(galaxy_id, params[i]),[t,pt])

bfm, bfa, bfbsc, bfrsc, bfbp1, bfrp1, bflm, bfz, bfd = dres.samples[-1]

np.save(pos_path + 'ALMA_{0}_bfit'.format(galaxy_id), [bfm, bfa, bfbsc, bfrsc, bfbp1, bfrp1, bflm, bfz, bfd, dres.logl[-1]])