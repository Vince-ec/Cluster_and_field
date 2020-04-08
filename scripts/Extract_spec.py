import numpy as np
import pandas as pd
from shutil import copyfile
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import fsps
from matplotlib import gridspec
import matplotlib as mpl
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import astropy.units as u
from sim_engine import Scale_model
from spec_tools import Source_present, Oldest_galaxy, Sig_int, Smooth, Rescale_sfh, lbt_to_z, boot_to_posterior, age_to_z, Posterior_spec
from spec_stats import Smooth, Highest_density_region, Linear_fit
from spec_id import *
from spec_stats import Highest_density_region, Linear_fit
from spec_exam import Gen_spec
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as sea
import os
from grizli import multifit
from grizli import model
from grizli.utils import SpectrumTemplate 
from sim_engine import forward_model_grism
import pickle
from spec_tools import Gen_SFH

alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
morph_db = alldb.query('W_UVJ == "Q" and AGN != "AGN" and lmass >= 10.5 and n_f < 3 and Re < 20 ')

bspec = [27458,294464,36348,48631,19290,32566,32691,33093,26272,35640,45333, 30144]
# nog141 = [27915,37955,17746,17735]
nog102 = [27714,37189,26139,32799,47223,22774,28890,23073,31452,24033]
# nog102 = []

inout = []
for i in morph_db.index:     
    if morph_db.id[i] not in bspec and morph_db.id[i] not in nog102: 
        inout.append('i')
    else:
        inout.append('o')
        
morph_db['inout'] = inout
mdb = morph_db.query('inout == "i" and 0.7 < zgrism < 2.5 and Sigma1 > 10**9.6')

def Extract_grism_fit(MB, fit, instr, lims):
    spec = fit['line1d'].wave, fit['line1d'].flux
    spf = fit['line1d'].wave, fit['line1d'].wave*0+1
    sptbl = MB.oned_spectrum(tfit = fit)

    W = sptbl[instr]['wave']

    fgrid = np.zeros([MB.N, len(W)])
    
    for i in range(MB.N):
        beam = MB.beams[i]
        b_mask = beam.fit_mask.reshape(beam.sh)

        m_i = beam.compute_model(spectrum_1d=spec, is_cgs=True, in_place=False).reshape(beam.sh)
        f_i = beam.compute_model(spectrum_1d=spf, is_cgs=True, in_place=False).reshape(beam.sh)

        grism = beam.grism.filter

        w, flm, erm = beam.beam.optimal_extract(m_i, bin=1, ivar=beam.ivar*b_mask)
        w, sens, ers = beam.beam.optimal_extract(f_i, bin=1, ivar=beam.ivar*b_mask)

        sens[~np.isfinite(sens)] = 1

        unit_corr = 1./sens

        clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1]]

        flm *= unit_corr
        
        FLUX = interp1d(w,flm, bounds_error=False, fill_value=0)(W)
        
        for ii in range(len(FLUX)):
            if not FLUX[ii]**2 > 0:
                FLUX[ii] = 0
        
        fgrid[i] = FLUX        
    
    w = sptbl[instr]['wave']
    f = sptbl[instr]['flux']
    e = sptbl[instr]['err']
    fl = sptbl[instr]['flat']
    m = []
    
    for ff in fgrid.T:
        m.append(np.mean(ff[ff>0]))
    clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1] and f[U]**2 > 0 and m[U] > 0]
    return w[clip], f[clip]/fl[clip], e[clip]/fl[clip], np.array(m)[clip]

temps = {}
for idx in mdb.index:
    Gs = Gen_spec_2D(mdb.field[idx], mdb.id[idx], mdb.zgrism[idx], g102_lims=[8300, 11288], 
                     g141_lims=[11288, 16500],phot_errterm = 0.04, irac_err = 0.08,) 

    wave, spec = np.load('../data/allsed/phot/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]))
    x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Pbp1.npy'.format(mdb.field[idx], mdb.id[idx]))
    bp1 = x[px == max(px)][0]
    x,px = np.load('../data/posteriors/{0}_{1}_tabfit_Prp1.npy'.format(mdb.field[idx], mdb.id[idx]))
    rp1 = x[px == max(px)][0]

    if Gs.g102:
        np.save('../data/allsed/g102/{}-{}_O'.format(mdb.field[idx], mdb.id[idx]),[Gs.Bwv, Gs.Bfl, Gs.Ber])
    if Gs.g141:
        np.save('../data/allsed/g141/{}-{}_O'.format(mdb.field[idx], mdb.id[idx]),[Gs.Rwv, Gs.Rfl, Gs.Rer])

    if Gs.g102:
        temps['fsps_model'] = SpectrumTemplate(wave=wave, flux=spec*1E18)
        fit = Gs.mb_g102.template_at_z(mdb.zgrism[idx], templates = temps, fitter='lstsq')
        w,f,e,m = Extract_grism_fit(Gs.mb_g102, fit, 'G102', [8300,11300])
        S1 = Scale_model(interp1d(wave*(1+mdb.zgrism[idx]),spec)(w), np.ones_like(w), m)
        S2 = Scale_model(f,e, m*S1)
        np.save('../data/allsed/g102/{}-{}'.format(mdb.field[idx], mdb.id[idx]),[w,f/S2,e/S2])
        np.save('../data/allsed/g102/{}-{}_mod'.format(mdb.field[idx], mdb.id[idx]),[w,m*S1])

    if Gs.g141:
        temps['fsps_model'] = SpectrumTemplate(wave=wave, flux=spec*1E18)
        fit = Gs.mb_g141.template_at_z(mdb.zgrism[idx], templates = temps, fitter='lstsq')
        w,f,e,m = Extract_grism_fit(Gs.mb_g141, fit, 'G141', [11300, 16500])
        S1 = Scale_model(interp1d(wave*(1+mdb.zgrism[idx]),spec)(w), np.ones_like(w), m)
        S2 = Scale_model(f,e, m*S1)
        np.save('../data/allsed/g141/{}-{}'.format(mdb.field[idx], mdb.id[idx]),[w,f/S2,e/S2])
        np.save('../data/allsed/g141/{}-{}_mod'.format(mdb.field[idx], mdb.id[idx]),[w,m*S1])