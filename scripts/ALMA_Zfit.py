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
from spec_id import Scale_model
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
    lim1 = int(sys.argv[1])
    lim2 = int(sys.argv[1])

Gs = Gen_ALMA_spec(galaxy_id, 1, g102_lims=[8750,11300], g141_lims=[lim1,lim2], mdl_err=False)

sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1),sfh = 0)

############
###priors###
def alma_prior(u):
    m = (0.03 * u[0] + 0.001) / 0.019
    a = (2)* u[1] + 1
    z = stats.norm.ppf(u[2],loc = 1.6, scale = 0.1)
    return [m, a, z]

############
#likelihood#
def Gather_grism_sim_data(spec):
    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    
    if spec.g102:
        wvs.append(spec.Bwv)
        flxs.append(spec.Bfl)
        errs.append(spec.Ber)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.Rfl)
        errs.append(spec.Rer)
        beams.append(spec.Rbeam)
        trans.append(spec.Rtrans)

    return np.array([wvs, flxs, errs, beams, trans])

def forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def Full_forward_model(spec, wave, flux, specz):
    Gmfl = []
    
    for i in range(len(wvs)):
        Gmfl.append(forward_model_all_beams(beams[i], trans[i], wvs[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl


def Full_fit(spec, Gmfl, Pmfl):
    Gchi = 0
    
    for i in range(len(wvs)):
        scale = Scale_model(flxs[i], errs[i], Gmfl[i])
        Gchi = Gchi + np.sum(((((flxs[i] / scale) - Gmfl[i]) / (errs[i] / scale))**2))

    Pchi = np.sum((((spec.Pflx - Pmfl) / spec.Perr)**2))
    
    return Gchi, Pchi

def Full_scale(spec, Pmfl):
    return Scale_model(spec.Pflx, spec.Perr, Pmfl)

wvs, flxs, errs, beams, trans = Gather_grism_sim_data(Gs)

def alma_L(X):
    m, a, z = X
    
    sp.params['logzsol'] = np.log10( m )
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)    
    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z) 
    PC = Full_scale(Gs, Pmfl)
    Gchi, Pchi = Full_fit(Gs, PC*Gmfl, PC*Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

############
####run#####
sampler = dynesty.DynamicNestedSampler(alma_L, alma_prior, ndim = 3, sample = 'rwalk', bound = 'single',
                                  queue_size = 8, pool = Pool(processes=8))  
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

dres = sampler.results
############
####save####
np.save(out_path + 'ALMA_{0}'.format(galaxy_id), dres) 

############# 
#get lightweighted age
#############

params = ['m', 'a', 'z']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + 'ALMA_{0}_P{1}'.format(galaxy_id, params[i]),[t,pt])

bfm, bfa, bfz = dres.samples[-1]

np.save(pos_path + 'ALMA_{0}_bfit'.format(galaxy_id), [bfm, bfa,bfz, dres.logl[-1]])