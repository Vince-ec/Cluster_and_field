__author__ = 'vestrada'

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from glob import glob
import os
from grizli import multifit
from grizli import model
from astropy.cosmology import Planck13 as cosmo
import fsps
from C_full_fit import Gen_mflgrid, Analyze_full_fit, Stich_grids,\
    Stitch_spec, Scale_model_mult, Resize
from time import time
from sim_engine import *

hpath = os.environ['HOME'] + '/'

"""
class:
Gen_spec

def:
"""

if hpath == '/home/vestrada78840/':
    from C_spec_tools import Source_present, Photometry, Scale_model, Oldest_galaxy
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/clear_q_beams/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/home/vestrada78840/chidat/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    from spec_tools import Source_present, Photometry, Scale_model, Oldest_galaxy
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/posteriors/'
    phot_path = '../phot/'

class Gen_spec(object):
    def __init__(self, field, galaxy_id, specz, g102_beam, g141_beam,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                tmp_err = True, phot_errterm = 0):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.set_scale = False
        self.sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, 
                                         zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)

        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        tmp_err - (flag) whether or not we apply a template error function
        """
        
        # load spec and phot
        self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB = load_spec(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz)
        
        self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR = load_spec(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz)
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False)
         
        self.Bfl = self.Bflx / self.Bflt 
        self.Rfl = self.Rflx / self.Rflt 
        
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
        
        ### set beams
        self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, g102_beam)
        self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, g141_beam)
        
        ### apply tmp_err         
        self.Berr = apply_tmp_err(self.Bwv_rf,self.Berr,self.Bflx, tmp_err = tmp_err)
        self.Rerr = apply_tmp_err(self.Rwv_rf,self.Rerr,self.Rflx, tmp_err = tmp_err)
        self.Perr = apply_tmp_err(self.Pwv_rf,self.Perr,self.Pflx, tmp_err = tmp_err, pht_err = phot_errterm)
        
        self.Ber = self.Berr / self.Bflt
        self.Rer = self.Rerr / self.Rflt
        
    def Sim_spec_mult(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return forward_model_grism(self.Bbeam, model_wave, model_flux), \
                forward_model_grism(self.Rbeam, model_wave, model_flux)

    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal / 0.019)
        self.sp.params['tau'] = tau
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        [Bmw, Bmf], [Rmw, Rmf] = self.Sim_spec_mult(model_wave * (1 + model_redshift), 
                                                    model_flux * Salmon(Av,model_wave))
        iBmf = interp1d(Bmw,Bmf)(self.Bwv)       
        iRmf = interp1d(Rmw,Rmf)(self.Rwv)     
        
        self.Bmfl = iBmf / self.Btrans
        self.Rmfl = iRmf / self.Rtrans
            
        self.Bmfl *= self.PC
        self.Rmfl *= self.PC
        
        if not self.set_scale:
            Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)
            Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

            self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
            self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
       
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal / 0.019)
        self.sp.params['tau'] = tau
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)
        
        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), 
                                                  model_flux * Salmon(Av,model_wave))
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self, model_wave, model_flux, bfz, bfd):      
        US_model_flux = F_lam_per_M(model_flux * Salmon(bfd,model_wave), model_wave * (1 + bfz), bfz, 0, 1)

        US_pfl = self.Sim_phot_mult(model_wave * (1 + bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        Bw,Bf = forward_model_grism(self.Bbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        Rw,Rf = forward_model_grism(self.Rbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        
        iBmf = interp1d(Bw,Bf)(self.Bwv)       
        iRmf = interp1d(Rw,Rf)(self.Rwv)  
  
        Bmfl = iBmf / self.Btrans
        Rmfl = iRmf / self.Rtrans

        self.Bscale = Scale_model(self.Bfl, self.Ber, Bmfl)
        self.Rscale = Scale_model(self.Rfl, self.Rer, Rmfl)
        
        self.Bfl = self.Bfl / self.Bscale ; self.Ber = self.SBer / self.Bscale 
        self.Rfl = self.Rfl / self.Rscale ; self.Rer = self.SRer / self.Rscale 
        
        self.set_scale = True
        
        
def Best_fitter(field, galaxy, g102_beam, g141_beam, specz,
                errterm = 0):
    ######## initialize spec
    
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, specz, g102_beam, g141_beam,
                   #g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm,)    
    
    metal_i = 0.019
    age_i = 2
    tau_i = 0.1
    rshift_i = specz
    dust_i = 0.1
    
    for x in range(3):
    
        metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)
    
        Bmfl, Rmfl, Pmfl = Gen_mflgrid(Gs, sp, metal, age, tau, rshift)

        ## set some variables
        [Bmwv,Bmflx], [Rmwv,Rmflx] = Gs.Sim_spec_mult(wave, flux)

        PC, Pgrid = Stitch_resize_redden_fit(Gs.Pwv, Gs.Pflx, Gs.Perr, Pmfl, Gs.Pwv, 
                         metal, age, tau, rshift, dust, phot = True) 
        Bgrid = Stitch_resize_redden_fit(Gs.Bwv, Gs.Bflx, Gs.Berr, Bmfl, Bmwv, 
                         metal, age, tau, rshift, dust, PC)
        Rgrid = Stitch_resize_redden_fit(Gs.Rwv, Gs.Rflx, Gs.Rerr, Rmfl, Rmwv, 
                         metal, age, tau, rshift, dust, PC)
        
        print('g102:', Best_fit_model(Bgrid, metal, age, tau, rshift, dust))
        print('g141:', Best_fit_model(Rgrid, metal, age, tau, rshift, dust))
        print('phot:', Best_fit_model(Pgrid, metal, age, tau, rshift, dust))
        
        
        
        
        bfd, bfZ, bft, bftau, bfz = Best_fit_model(Pgrid + Bgrid +Rgrid, metal, age, tau, rshift, dust)
        
        metal_i = bfZ
        age_i = bft
        tau_i = bftau
        rshift_i = bfz
        dust_i = bfd
        
        print(bfZ, bft, bftau, specz, bfd)   

def Best_fit_model(chi, metal, age, tau, rshift, dust):
    x = np.argwhere(chi == np.min(chi))[0]
    return dust[x[0]],metal[x[1]], age[x[2]], tau[x[3]] , rshift[x[4]]
        
def Stitch_resize_redden_fit(fit_wv, fit_fl, fit_er, mfl, mwv, 
                     metal, age, tau, rshift, dust, PC = 0, phot=False):
    #############Read in spectra and stich spectra grid together#################
    if phot:
        PC, chigrid = Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust, phot = True)  
        return PC, chigrid
    else:
        mfl = Resize(fit_wv, mwv, mfl)
        chigrid = Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust, PC)  
        return chigrid
        
def Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, Av, PC = 0, phot = False):    
    minidust = Gen_dust_minigrid(fit_wv, redshift, Av)

    scales = []
    chigrids = []
    
    if phot:
        for i in range(len(Av)):
            dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
                [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
            redflgrid = mfl * dustgrid        
            SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
            redflgrid = np.array([SCL]).T*redflgrid
            chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
        
            scales.append(np.array([SCL]).T)
            chigrids.append(chigrid)
            
        return np.array(scales), np.array(chigrids) 

    else:
        for i in range(len(Av)):
            dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
                [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
            redflgrid = mfl * dustgrid
            redflgrid = PC[i]*redflgrid
            SCL2 = Scale_model_mult(fit_fl,fit_er,redflgrid)
            chigrid = np.sum(((fit_fl / np.array([SCL2]).T - redflgrid) / (fit_er / np.array([SCL2]).T)) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
            chigrids.append(chigrid)

        return np.array(chigrids)

def Gen_dust_minigrid(fit_wv, rshift, Av):
    dust_dict = {}
    for i in range(len(Av)):
        key = str(Av[i])
        minigrid = np.zeros([len(rshift),len(fit_wv)])
        for ii in range(len(rshift)):
            minigrid[ii] = Salmon(Av[i],fit_wv / (1 + rshift[ii]))
        dust_dict[key] = minigrid
    return dust_dict

def Resize(fit_wv, mwv, mfl):
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(mwv,range(len(mfl.data)),mfl.data)(fit_wv,range(len(mfl.data)))
    return mfl

def Gen_mflgrid(spec, models, metal, age, tau, rshift):
    wv,fl = models.get_spectrum(tage = 2.0, peraa = True)
    [Bmwv,Bmf_len], [Rmwv,Rmf_len] = spec.Sim_spec_mult(wv,fl)
    
    ##### set model wave
    
    Bmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(Bmf_len)])
    Rmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(Rmf_len)])
    Pmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(spec.IDP)])
    
    for i in range(len(metal)):
        models.params['logzsol'] = np.log10(metal[i] / 0.019)
        for ii in range(len(age)):
            for iii in range(len(tau)):
                models.params['tau'] = tau[iii]
                wv,fl = models.get_spectrum(tage = age[ii], peraa = True)
                for iv in range(len(rshift)):
                    [Bmwv,Bmflx], [Rmwv,Rmflx] = spec.Sim_spec_mult(wv * (1 + rshift[iv]),fl)
                    Pmflx = spec.Sim_phot_mult(wv * (1 + rshift[iv]),fl)

                    Bmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Bmflx
                    Rmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Rmflx
                    Pmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Pmflx
    
    return Bmfl, Rmfl, Pmfl
            
def Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, stage):
    if stage == 0:
        age = np.round(np.arange(0.5, 6.1, .25),2)
        metal= np.round(np.arange(0.002 , 0.031, 0.003),4)
        tau = np.round(np.logspace(np.log10(0.01), np.log10(3), 8), 3)
        rshift = np.round(np.arange( rshift_i - 0.01, rshift_i + 0.011, 0.002),4)
        dust = np.round(np.arange(0, 1.1, 0.1),2)
    
    if stage == 1:
        if age_i <=1.5:
            age_i = 1.6
        age = np.round(np.arange(age_i  - 1.5, age_i  + 1.6, .125),2)
        
        if metal_i <= 0.0075:
            metal_i = 0.0095
        metal= np.round(np.arange(metal_i  - 0.0075, metal_i  + 0.0085, 0.0015),4)
        
        if tau_i <= 1:
            tau_i = 1.1 
        tau = np.round(np.logspace(np.log10(tau_i  - 1), np.log10(tau_i  + 1), 8), 3)
        rshift = np.round(np.arange( rshift_i - 0.005, rshift_i + 0.006, 0.001),4)
        
        if dust_i <= 0.25:
            dust_i = 0.25         
        dust = np.round(np.arange(dust_i - 0.25, dust_i + 0.3, 0.05),2)
    
    if stage == 2:
        if age_i <= 0.75:
            age_i = 0.85
        age = np.round(np.arange(age_i  - 0.75, age_i  + 0.85, .06),2)
        
        if metal_i <= 0.00375:
            metal_i = 0.00575
        metal= np.round(np.arange(metal_i  - 0.00375, metal_i  + 0.00475, 0.00075),4)
                
        if tau_i <= 0.5:
            tau_i = 0.51   
        tau = np.round(np.logspace(np.log10(tau_i  - 0.5), np.log10(tau_i  + 0.5), 8), 3)
        rshift = np.round(np.arange( rshift_i - 0.0025, rshift_i + 0.003, 0.0005),4)
        
        if dust_i <= 0.125:
            dust_i = 0.125    
        dust = np.round(np.arange(dust_i - 0.125, dust_i + 0.135, 0.025),3)
    
    return metal, age, tau, rshift, dust