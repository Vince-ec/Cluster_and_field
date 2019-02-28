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
    name = sys.argv[1]
    tau = sys.argv[2]
    trim = float(sys.argv[3])
    
if tau == 'tab':
    sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 3, dust_type = 1)

if tau == 'delay':
    sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau = 0.1, dust_type = 1)
    
Gs = Gen_spec('GND', 21156, 1.253,
               g102_lims=[8300, 11500], g141_lims=[11100, 16500], mdl_err = True,
            phot_errterm = 0.03, decontam = True, trim = trim) 

############
###priors###
agelim = Oldest_galaxy(1.253 + .1)
     
if tau == 'tab':
    
    def prior_transform(u):
        m = (0.03 * u[0] + 0.001) / 0.019
        a = agelim * u[1] + 0.1

        t1 = u[2]
        t2 = u[3]
        t3 = u[4]
        t4 = u[5]
        t5 = u[6]
        z = 1.253 + 0.1*(2*u[7] - 1)
        d = u[8]

        return [m, a, t1, t2, t3, t4, t5, z, d]

if tau == 'delay':
    
    def prior_transform(u):
        m = (0.03 * u[0] + 0.001) / 0.019
        a = agelim * u[1] + 0.1
        t = u[2]
        z = 1.253 + 0.1*(2*u[3] - 1)
        d = u[4]

        return [m, a, t, z, d]
    
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

if tau == 'tab':
    def loglikelihood(X):
        m, a, t1, t2, t3, t4, t5, z, d = X

        sp.params['logzsol'] = np.log10( m )
        sp.params['dust2'] = d
        sp.set_tabular_sfh(np.array([0.25, 0.75, 1.5, 3, 7]),np.array([t1, t2, t3, t4, t5]))
        wave, flux = sp.get_spectrum(tage = a, peraa = True)

        Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z)

        PC= Full_scale(Gs, Pmfl)

        Gchi, Pchi = Full_fit(Gs, PC * Gmfl, PC * Pmfl)

        return -0.5 * (Gchi + Pchi)
    
if tau == 'delay':
    def loglikelihood(X):
        m, a, t, z, d = X

        sp.params['logzsol'] = np.log10( m )
        sp.params['dust2'] = d
        sp.params['tau'] = t
        wave, flux = sp.get_spectrum(tage = a, peraa = True)

        Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z)

        PC= Full_scale(Gs, Pmfl)

        Gchi, Pchi = Full_fit(Gs, PC * Gmfl, PC * Pmfl)

        return -0.5 * (Gchi + Pchi)
    
############
####run#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

if tau == 'tab':
    dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim = 9, sample = 'rwalk', bound = 'balls') 

if tau == 'delay':
    dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim = 5, sample = 'rwalk', bound = 'balls') 

dsampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=False)

dres = dsampler.results
############
####save####
np.save(out_path + 'GND_21156_{0}_{1}_{2}_testfit.npy'.format(tau, name, trim), dres) 
