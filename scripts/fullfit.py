#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_exam import Gen_spec
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
import fsps
import dynesty
from scipy.interpolate import interp1d, RegularGridInterpolator
from sim_engine import forward_model_grism, Salmon
from spec_id import Scale_model
from spec_tools import Oldest_galaxy

hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/home/vestrada78840/chidat/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/posteriors/'
    phot_path = '../phot/'
    
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = sys.argv[2] 
    specz = float(sys.argv[3])
    
sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 3, dust_type = 1)

Gs = Gen_spec(field, galaxy, 1, g102_lims=[8300, 11288], g141_lims=[11288, 16500], mdl_err = True,
            phot_errterm = 0.0, decontam = True) 

############
###priors###
agelim = Oldest_galaxy(specz - 0.01)
     
def prior_transform(u):
    m = (0.03 * u[0] + 0.001) / 0.019
    a = agelim * u[1] + 0.01

    t1 = u[2]
    t2 = u[3]
    t3 = u[4]
    t4 = u[5]
    t5 = u[6]
    t6 = u[7]
    
    z = specz + 0.01*(2*u[8] - 1)
    d = 2*u[9]
    
    return [m, a, t1, t2, t3, t4, t5, t6, z, d]


############
#likelihood#
def forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def Full_scale(spec, Pmfl):
    return Scale_model(spec.Pflx, spec.Perr, Pmfl)

def Gather_grism_data(spec):
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

def Time_bins(agelim):
    lbt = np.array([0, 0.1, 0.3, 0.6, 1, 1.5])
    return np.round(agelim  - lbt / 2.1 * agelim, 2)[::-1]

LBT = Time_bins(agelim)

def loglikelihood(X):
    m, a, t1, t2, t3, t4, t5, t6, z, d = X
    
    sp.params['logzsol'] = np.log10( m )
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    
    sp.set_tabular_sfh(LBT, np.array([t1, t2, t3, t4, t5, t6]))
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)
    
    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z)
    
    PC= Full_scale(Gs, Pmfl)

    Gchi, Pchi = Full_fit(Gs, PC * Gmfl, PC * Pmfl)
                  
    return -0.5 * (Gchi + Pchi)
############
####run#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim = 10, sample = 'rwalk', bound = 'balls') 
dsampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=False)

dres = dsampler.results
############
####save####
np.save(out_path + '{0}_{1}_nestfit.npy'.format(field, galaxy), dres) 
