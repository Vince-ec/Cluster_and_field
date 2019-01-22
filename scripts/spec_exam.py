__author__ = 'vestrada'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from matplotlib import gridspec
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
                mdl_err = True, instr_err = True, phot_errterm = 0, decontam = False):
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
        try:
            self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = load_spec(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz)
            if decontam:
                self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = decontaminate(self.Bwv, 
                        self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont)
                print('cleaned')
            self.Bfl = self.Bflx / self.Bflt 
            self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, g102_beam)
            self.Berr = apply_tmp_err(self.Bwv, self.Bwv_rf, self.Berr, self.Bflx, 'B', mdl_err = mdl_err, instr_err = instr_err)
            self.Ber = self.Berr / self.Bflt
            self.g102 = True

        except:
            print('missing g102')
            self.g102 = False
        
        try:
            self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = load_spec(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz)
            
            if decontam:
                self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = decontaminate(self.Rwv, 
                                self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont)
                
            self.Rfl = self.Rflx / self.Rflt 
            self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, g141_beam)
            self.Rerr = apply_tmp_err(self.Rwv, self.Rwv_rf, self.Rerr, self.Rflx, 'R', mdl_err = mdl_err, instr_err = instr_err)
            self.Rer = self.Rerr / self.Rflt
            self.g141 = True

        except:
            print('missing g141')
            self.g141 = False
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False)
         
        
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
               
        ### apply tmp_err         
        self.Perr = apply_tmp_err(self.Pwv, self.Pwv_rf,self.Perr,self.Pflx, 'P', mdl_err = mdl_err,
                                  instr_err = instr_err , pht_err = phot_errterm)

    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal / 0.019)
        self.sp.params['tau'] = tau
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        if self.g102:
        
            Bmw, Bmf= forward_model_grism(self.Bbeam, model_wave * (1 + model_redshift), 
                                                        model_flux * Salmon(Av,model_wave))
            iBmf = interp1d(Bmw,Bmf)(self.Bwv)       

            self.Bmfl = iBmf / self.Btrans

            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            Rmw, Rmf = forward_model_grism(self.Rbeam, model_wave * (1 + model_redshift), 
                                                        model_flux * Salmon(Av,model_wave))
            iRmf = interp1d(Rmw,Rmf)(self.Rwv)     

            self.Rmfl = iRmf / self.Rtrans

            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

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
        
    def Scale_flux(self,  bfZ, bft, bftau, bfz, bfd):      
        self.sp.params['logzsol'] = np.log10(bfZ / 0.019)
        self.sp.params['tau'] = bftau
        model_wave,model_flux = self.sp.get_spectrum(tage = bft, peraa = True)
        
        US_model_flux = F_lam_per_M(model_flux * Salmon(bfd,model_wave), model_wave * (1 + bfz), bfz, 0, 1)

        US_pfl = self.Sim_phot_mult(model_wave * (1 + bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        if self.g102:  
            Bw,Bf = forward_model_grism(self.Bbeam, self.model_wave * (1 + bfz), self.S_model_flux)
            iBmf = interp1d(Bw,Bf)(self.Bwv)       
            Bmfl = iBmf / self.Btrans
            self.Bscale = Scale_model(self.Bfl, self.Ber, Bmfl)
            self.Bfl = self.Bfl / self.Bscale ; self.Ber = self.Ber / self.Bscale 
        
        if self.g141:    
            Rw,Rf = forward_model_grism(self.Rbeam, self.model_wave * (1 + bfz), self.S_model_flux)
            iRmf = interp1d(Rw,Rf)(self.Rwv)  
            Rmfl = iRmf / self.Rtrans
            self.Rscale = Scale_model(self.Rfl, self.Rer, Rmfl)
            self.Rfl = self.Rfl / self.Rscale ; self.Rer = self.Rer / self.Rscale 
        
        self.set_scale = True
        
    def Make_sim(self,  bfZ, bft, bftau, bfz, bfd):
        self.sp.params['logzsol'] = np.log10(bfZ / 0.019)
        self.sp.params['tau'] = bftau
        model_wave,model_flux = self.sp.get_spectrum(tage = bft, peraa = True)
        
        
        ### set sim and transmission curve
        self.SBflx, self.SBerr, self.SBfl, self.SBer, self.SRflx, self.SRerr, self.SRfl, self.SRer, \
            self.SPflx, self.SPerr, self.mass, self.logmass =  init_sim(model_wave, 
            model_flux * Salmon(bfd,model_wave), bfz, self.sp.stellar_mass, self.Bwv, self.Rwv, 
            self.Bflx, self.Rflx, self.Pflx, self.Berr, self.Rerr, self.Perr, 0, 
            self.Btrans, self.Rtrans, self.Bflt, self.Rflt, self.Bbeam, self.Rbeam, 
            self.IDP, self.sens_wv, self.b, self.dnu, self.adj)