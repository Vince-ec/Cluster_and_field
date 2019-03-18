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
from astropy.cosmology import Planck13 as cosmo
from multiprocessing import Pool

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
    
specz = 1.25

sim2 = Gen_spec('GND', 21156, 1.25257,
               g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = True,
            phot_errterm = 0.0, decontam = False) 

sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau=0.1, dust_type = 1)
sp.params['dust2'] =0.2
sp.params['dust1'] =0.2
sp.params['tau'] =0.3
sp.params['logzsol'] = np.log10(0.8)

wave2, flux2 = sp.get_spectrum(tage = 3.5, peraa = True)

mass_perc2 = sp.stellar_mass

D_l = cosmo.luminosity_distance(specz).value # in Mpc
conv = 3.086E24
lsol_to_fsol = 3.839E33

sim2.Make_sim(wave2, flux2 * 10**11* lsol_to_fsol / (4 * np.pi * (D_l*conv)**2), specz)

def Time_bins(agelim, bins):
    u = 0.0
    lbt = []
    for i in range(bins):
        u+=0.1 * i
        lbt.append(np.round(u,1))
    
    return np.array(agelim  - lbt / np.round(u + 0.1 * (i+1),1) * agelim)[::-1]

LBT = Time_bins(Oldest_galaxy(1.25),10)

sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 3, sfh = 3, dust_type = 1)

############
###priors###
agelim = Oldest_galaxy(specz)

def tab_prior(u):
    m1 = (0.03 * u[0] + 0.001) / 0.019
    m2 = (0.03 * u[1] + 0.001) / 0.019
    m3 = (0.03 * u[2] + 0.001) / 0.019
    m4 = (0.03 * u[3] + 0.001) / 0.019
    m5 = (0.03 * u[4] + 0.001) / 0.019
    m6 = (0.03 * u[5] + 0.001) / 0.019
    m7 = (0.03 * u[6] + 0.001) / 0.019
    m8 = (0.03 * u[7] + 0.001) / 0.019
    m9 = (0.03 * u[8] + 0.001) / 0.019
    m10 = (0.03 * u[9] + 0.001) / 0.019
    
    a = (agelim - LBT[0])* u[10] + LBT[0]
    
    t1 = u[11]
    t2 = u[12]
    t3 = u[13]
    t4 = u[14]
    t5 = u[15]  
    t6 = u[16]
    t7 = u[17]
    t8 = u[18]
    t9 = u[19]
    t10 = u[20] 
    
    z = specz + 0.002*(2*u[21] - 1)
    
    d = 1*u[22]
    
    lm = 11.0 + 1.25*(2*u[23] - 1)
    
    return [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, a, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, z, d, lm]

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
        flxs.append(spec.SBfl)
        errs.append(spec.SBer)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.SRfl)
        errs.append(spec.SRer)
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
    
    for i in range(len(wvs2)):
        Gmfl.append(forward_model_all_beams(beams2[i], trans2[i], wvs2[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl


def Full_fit(spec, Gmfl, Pmfl):
    Gchi = 0
    
    for i in range(len(wvs2)):
        scale = Scale_model(flxs2[i], errs2[i], Gmfl[i])
        Gchi = Gchi + np.sum(((((flxs2[i] / scale) - Gmfl[i]) / (errs2[i] / scale))**2))
    
    Pchi = np.sum((((spec.SPflx - Pmfl) / spec.SPerr)**2))
    
    return Gchi, Pchi

wvs2, flxs2, errs2, beams2, trans2 = Gather_grism_sim_data(sim2)

def tab_L(X):
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, a, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, z, d, lm = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    
    sp.set_tabular_sfh(LBT,np.array([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]),
                      Z = np.array([m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]) * 0.019)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)
    
    D_l = cosmo.luminosity_distance(z).value # in Mpc

    mass_transform = (10**lm / sp.stellar_mass) * lsol_to_fsol / (4 * np.pi * (D_l*conv)**2)  

    Gmfl, Pmfl = Full_forward_model(sim2, wave, flux * mass_transform, z)
      
    Gchi, Pchi = Full_fit(sim2, Gmfl, Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

############
####run#####
d_tsampler = dynesty.NestedSampler(tab_L, tab_prior, ndim = 24, sample = 'rwalk', bound = 'balls',
                                  queue_size = 8, pool = Pool(processes=8))  
d_tsampler.run_nested(print_progress=False)

dres = d_tsampler.results
############
####save####
np.save(out_path + 'sim_test_delay_to_tab_{0}'.format(runnum), dres) 
