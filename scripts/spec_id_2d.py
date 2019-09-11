import numpy as np
import pandas as pd
import os
from glob import glob

from spec_id import convert_sfh, get_agebins

import fsps
from grizli import multifit
from grizli import model
from grizli.utils import SpectrumTemplate

from time import time

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
    args = np.load('/fdata/scratch/vestrada78840/multifit_data/fit_args.npy')[0]

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
    args = np.load('../data/multifit_data/fit_args.npy')[0]

def Gen_multibeams(beams, args = args):
    mb = multifit.MultiBeam(beams,**args)

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
    
    return mb_g102, mb_g141

def Gen_model(sp, params, masses, agebins = 10, SF = False):
    #### make sure fsps is initialized correctly####
    """
    return a spectrum
    
    sp : fsps object
    params : should be ordered as metallicity (Z), age (a), Av (d)
    masses : masses used to create SFH
    agebins : how many bins used for SFH
    SF : boolean, if SF then dust1 = 0
    SFH : boolean, whether or not to return SFH
    """
    Z, a, d = params

    if SF:
        sp.params['dust1'] = 0 
    else:
        sp.params['dust1'] = d
        
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(Z)

    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = agebins), masses , maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    return wave, flux
     
def Gen_temp_dict(redshift, lowlim, hilim, args = args):
    temps = {}
    
    for k in args['t1']:
        if k[0] == 'l' and lowlim < args['t1'][k].wave[args['t1'][k].flux == args['t1'][k].flux.max()][0] * (1+redshift) < hilim:
            temps[k] = args['t1'][k]
                
    return temps
 
def Gen_temp_dict_balm(redshift, lowlim, hilim, lines, args = args):
    temps = {}
    
    balmer = []
    
    for ln in lines:
        balmer.append('line ' + ln)
        
    for k in args['t1']:
        if k in balmer and lowlim < args['t1'][k].wave[args['t1'][k].flux == args['t1'][k].flux.max()][0] * (1+redshift) < hilim:
            temps[k] = args['t1'][k]
                
    return temps

def spec_construct(g102_fit,g141_fit, z, wave0 = 4000, usetilt = True):
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