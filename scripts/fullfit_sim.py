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
    runnum = sys.argv[1] 

tau_range = np.load(template_path + 'tau_range.npy')
metal_range = np.load(template_path + 'metal_range.npy')
age_range = np.load(template_path + 'age_range.npy')
lwagrid = np.load(template_path + 'lwa_grid.npy')

ilwagrid = RegularGridInterpolator([metal_range,tau_range],lwagrid)
    
sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau = 0.1)

Gs = Gen_spec('GND', 21156, 1.2529, beam_path + 'o151.0_21156.g102.A.fits',  beam_path + 'o144.0_21156.g141.A.fits',
               g102_lims=[7900, 11300], g141_lims=[11100, 16500],
            phot_errterm = 0.03, decontam = False)  

Gs.Make_sim(0.019, 3.2, 0.2 , 1.2, 0)

############
###priors###
specz = 1.2
bft = 0.2
bfd = 0

if bft <= 0.5:
    tau_limit = 0.5
    
if 0.5 < bft <= 1.0:
    tau_limit = 1.0
    
if bft > 1.0:
    tau_limit = 2.0
    
if bfd <= 0.5:
    dust_limit = 0.5
    
if 0.5 < bfd <= 1.0:
    dust_limit = 1.0
    
if bfd > 1.0:
    dust_limit = 2.0

def prior_transform(u):
    m = (0.03 * u[0] + 0.001) / 0.019
    a = 5. * u[1] + 0.1
    t = tau_limit*u[2] + 0.01
    z = specz + 0.004 * (2*u[3] - 1)
    d = dust_limit*u[4]
    
    lwvs = ilwagrid([m,t])[0]

    lwa = interp1d(age_range,lwvs)(a)
        
    return [m, lwa, t, z, d]
############
#likelihood#
def Full_forward_model(spec, wave, flux, specz):
    Bmwv, Bmflx= forward_model_grism(spec.Bbeam, wave * (1 + specz), flux)
    Rmwv, Rmflx= forward_model_grism(spec.Rbeam, wave * (1 + specz), flux)
    
    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)
    
    Bmfl = Resize(spec.Bwv,Bmwv,Bmflx)
    Rmfl = Resize(spec.Rwv,Rmwv,Rmflx)
    
    return Bmfl, Rmfl, Pmfl

def Full_scale(spec, Bmfl, Rmfl, Pmfl):

    BC = Scale_model(spec.SBflx, spec.SBerr, Bmfl)
    RC = Scale_model(spec.SRflx, spec.SRerr, Rmfl)
    PC = Scale_model(spec.SPflx, spec.SPerr, Pmfl)

    return BC, RC, PC


def Full_fit(spec, Bmfl, Rmfl, Pmfl):

    Bchi = np.sum((((spec.SBflx - Bmfl) / spec.SBerr)**2))
    Rchi = np.sum((((spec.SRflx - Rmfl) / spec.SRerr)**2))
    Pchi = np.sum((((spec.SPflx - Pmfl) / spec.SPerr)**2))
    
    return Bchi, Rchi, Pchi

def Resize(fit_wv, mwv, mfl):
    mfl = interp1d(mwv,mfl)(fit_wv)
    return mfl


def loglikelihood(X):
    m,lwa,t,z,d = X
    
    sp.params['logzsol'] = np.log10( m )
    sp.params['tau'] = t
    
    lwvs = ilwagrid([m,t])[0]

    a = interp1d(lwvs,age_range)(lwa)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)
        
    Bmfl, Rmfl, Pmfl = Full_forward_model(Gs, wave, flux * Salmon(d, wave), z)
    
    BC, RC, PC = Full_scale(Gs, Bmfl, Rmfl, Pmfl)

    Bchi, Rchi, Pchi = Full_fit(Gs, BC * Bmfl, RC * Rmfl, PC * Pmfl)
                  
    return -0.5 * (Bchi + Rchi + Pchi)
############
####run#####
dsampler = dynesty.DynamicNestedSampler(loglikelihood, prior_transform, ndim = 5, sample = 'rwalk') 

dsampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.005, print_progress=False)


dres = dsampler.results
############
####save####
np.save(out_path + 'sim_nestedfit_{0}.npy'.format(runnum), dres) 
