__author__ = 'vestrada'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from glob import glob
import os
from grizli import multifit
from grizli import model
from astropy.cosmology import Planck13 as cosmo
import fsps
#from spec_tools import Scale_model
from time import time
from sim_engine import *
from matplotlib import gridspec
hpath = os.environ['HOME'] + '/'

"""
class:
Gen_spec
Gen_ALMA_spec
Gen_SF_spec

def:
"""

if hpath == '/home/vestrada78840/':
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    beam_2d_path = '/fdata/scratch/vestrada78840/beams/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/home/vestrada78840/chidat/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    beam_2d_path = '/Volumes/Vince_CLEAR/RELEASE_v2.1.0/BEAMS/'
    template_path = '../templates/'
    out_path = '../data/posteriors/'
    phot_path = '../phot/'


class Gen_spec(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8300, 11288], g141_lims=[11288, 16500],
                phot_errterm = 0, irac_err = None, decontam = True, Bselect = None, Rselect = None, auto_select = False):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.set_scale = False
        self.g102_beam = glob(beam_path + '*{0}*g102*'.format(galaxy_id))
        self.g141_beam = glob(beam_path + '*{0}*g141*'.format(galaxy_id))
        self.sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau = 0.1, dust_type = 1)

        if len(self.g102_beam) < 1 :
            self.g102_beam = glob(beam_path + '*{0}*g102*'.format(39170))
        if len(self.g141_beam) < 1 :
            self.g141_beam = glob(beam_path + '*{0}*g141*'.format(39170))
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
        
         
        ##load spec and phot
        try:
            self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = load_spec(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz, 
                                select = Bselect, auto_select = auto_select, decontam = decontam)
            self.Bfl = self.Bflx / self.Bflt 
            self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, self.field, self.galaxy_id, 'G102')

            self.Ber = self.Berr / self.Bflt
            self.g102 = True

        except:
            print('missing g102')
            self.g102 = False
        
        try:
            self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = load_spec(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz, 
                                select = Rselect, auto_select = auto_select, decontam = decontam)
            self.Rfl = self.Rflx / self.Rflt 
            self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, self.field, self.galaxy_id, 'G141')

            self.Rer = self.Rerr / self.Rflt
            self.g141 = True

        except:
            print('missing g141')
            self.g141 = False
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False, select = None)
         
        self.Perr = apply_phot_err(self.Pflx, self.Perr, self.Pnum, base_err = phot_errterm, irac_err = irac_err)
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
               
    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal)
        self.sp.params['tau'] = tau
        self.sp.params['dust2'] = Av
        
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam, self.Btrans, self.Bwv, model_wave * (1 + model_redshift), 
                                                        model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam, self.Rtrans, self.Rwv, model_wave * (1 + model_redshift), 
                                                        model_flux) 
            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

                self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
    
    def Sim_spec_premade(self, model_wave, model_flux):
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam, self.Btrans, self.Bwv, model_wave, model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam, self.Rtrans, self.Rwv, model_wave, model_flux) 
            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

                self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
    
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal)
        self.sp.params['tau'] = tau
        self.sp.params['dust2'] = Av
        
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_phot_premade(self, model_wave, model_flux, scale = True):
        self.Pmfl = self.Sim_phot_mult(model_wave, model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)
        
        if scale == False:
            self.PC = 1
            
        self.Pmfl = self.Pmfl * self.PC
        
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self,  bfZ, bft, bftau, bfz, bfd):      
        self.sp.params['logzsol'] = np.log10(bfZ)
        self.sp.params['tau'] = bftau
        self.sp.params['dust2'] = bfd
        
        model_wave,model_flux = self.sp.get_spectrum(tage = bft, peraa = True)
        
        US_model_flux = F_lam_per_M(model_flux, model_wave * (1+bfz), bfz, 0, self.sp.stellar_mass)

        US_pfl = self.Sim_phot_mult(model_wave * (1+bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        if self.g102:  
            Bmf= self.Forward_model_all_beams(self.Bbeam, self.Bwv, model_wave * (1+bfz), 
                                                        self.S_model_flux)       
            Bmfl = Bmf / self.Btrans
            self.Bscale = Scale_model(self.Bfl, self.Ber, Bmfl)
            self.Bfl = self.Bfl / self.Bscale ; self.Ber = self.Ber / self.Bscale 
        
        if self.g141:    
            Rmf= self.Forward_model_all_beams(self.Rbeam, self.Rwv, model_wave * (1+bfz), 
                                                        self.S_model_flux) 
            Rmfl = Rmf / self.Rtrans
            self.Rscale = Scale_model(self.Rfl, self.Rer, Rmfl)
            self.Rfl = self.Rfl / self.Rscale ; self.Rer = self.Rer / self.Rscale 
        
        self.set_scale = True
        
    def Make_sim(self, model_wave, model_flux, specz, rndstate = 10, perturb = True):       
        self.SBfl, self.SBer,  self.SRfl, self.SRer, self.SPflx, self.SPerr =  init_sim(model_wave, 
            model_flux, specz, self.Bwv, self.Rwv, 
            self.Bfl, self.Rfl, self.Pflx, self.Ber, self.Rer, self.Perr, 0, 
            self.Btrans, self.Rtrans, self.Bbeam, self.Rbeam, 
            self.IDP, self.sens_wv, self.b, self.dnu, self.adj , rndstate = rndstate, perturb = perturb)
        
    @staticmethod
    def Forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
        return forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux)
    
    def Sim_all_premade(self, model_wave, model_flux, scale = True):
        self.Sim_phot_premade(model_wave, model_flux, scale = scale)
        self.Sim_spec_premade(model_wave, model_flux)

        
    def Full_forward_model(self, model_wave, model_flux, specz):
        
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam , self.Btrans, self.Bwv, model_wave * (1 + specz), model_flux)

        if self.g141:
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam , self.Rtrans, self.Rwv, model_wave * (1 + specz), model_flux)
            
        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + specz), model_flux)
        
    def Best_fit_scale(self, model_wave, model_flux, specz, bp1, rp1, massperc,logmass):
        self.Full_forward_model(model_wave, F_lam_per_M(model_flux, model_wave * (1 + specz),
                            specz, 0, massperc)*10**logmass, specz)

        if self.g102:
            self.bcal = Calibrate_grism([self.Bwv, self.Bfl, self.Ber], self.Bmfl, bp1)
            self.bscale = Scale_model(self.Bfl / self.bcal, self.Ber/ self.bcal, self.Bmfl)
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
            
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)               
            self.rscale = Scale_model(self.Rfl / self.rcal, self.Rer/ self.rcal, self.Rmfl)
            self.Rfl =  self.Rfl/ self.rcal/ self.rscale
            self.Rer =  self.Rer/ self.rcal/ self.rscale
        
    def Best_fit_scale_flam(self, model_wave, model_flam, specz, bp1, rp1):
        self.Full_forward_model(model_wave, model_flam, specz)

        if self.g102:
            self.bcal = Calibrate_grism([self.Bwv, self.Bfl, self.Ber], self.Bmfl, bp1)
            self.bscale = Scale_model(self.Bfl / self.bcal, self.Ber/ self.bcal, self.Bmfl)
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
            
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)               
            self.rscale = Scale_model(self.Rfl / self.rcal, self.Rer/ self.rcal, self.Rmfl)
            self.Rfl =  self.Rfl/ self.rcal/ self.rscale
            self.Rer =  self.Rer/ self.rcal/ self.rscale
        
###########
###########
        
class Gen_ALMA_spec(object):
    def __init__(self, galaxy_id, specz,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                mdl_err = True, phot_errterm = 0, decontam = False, trim = None):
        self.field = 'GSD'
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.set_scale = False
        self.g102_beam = glob(beam_path + '*{0}*g102*'.format(41520))
        self.g141_beam = glob(beam_path + '*{0}*g141*'.format(41520))
        self.sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), 
                                sfh = 4, tau = 0.1, dust_type = 1)

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
            self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = load_ALMA_spec(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz, trim = trim)
            if decontam:
                self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = decontaminate(self.Bwv, 
                        self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont)
                print('cleaned')
            self.Bfl = self.Bflx / self.Bflt 
            self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, self.g102_beam)
            self.Berr = apply_tmp_err(self.Bwv, self.Bwv_rf, self.Berr, self.Bflx, 'B', mdl_err = mdl_err)
            self.Ber = self.Berr / self.Bflt
            self.g102 = True

        except:
            print('missing g102')
            self.g102 = False
        
        try:
            self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = load_ALMA_spec(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz, trim = trim)

            if decontam:
                self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = decontaminate(self.Rwv, 
                                self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont)

            self.Rfl = self.Rflx / self.Rflt 
            self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, self.g141_beam)
            self.Rerr = apply_tmp_err(self.Rwv, self.Rwv_rf, self.Rerr, self.Rflx, 'R', mdl_err = mdl_err)
            self.Rer = self.Rerr / self.Rflt
            self.g141 = True

        except:
            print('missing g141')
            self.g141 = False
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_ALMA_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False, trim = trim)
         
        self.Perr = np.sqrt(self.Perr**2 + (phot_errterm*self.Pflx)**2)
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
               
        ### apply tmp_err         
        self.Perr = apply_tmp_err(self.Pwv, self.Pwv_rf,self.Perr,self.Pflx, 'P', mdl_err = mdl_err)

    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal)
        self.sp.params['tau'] = tau
        self.sp.params['dust2'] = Av
        
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam, self.Btrans, self.Bwv, model_wave * (1 + model_redshift), 
                                                        model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam, self.Rtrans, self.Rwv, model_wave * (1 + model_redshift), 
                                                        model_flux) 
            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

                self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
    
    def Sim_spec_premade(self, model_wave, model_flux):
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam, self.Btrans, self.Bwv, model_wave, model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam, self.Rtrans, self.Rwv, model_wave, model_flux) 
            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

                self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
    
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        self.sp.params['logzsol'] = np.log10(metal)
        self.sp.params['tau'] = tau
        self.sp.params['dust2'] = Av
        
        model_wave,model_flux = self.sp.get_spectrum(tage = age, peraa = True)

        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_phot_premade(self, model_wave, model_flux, scale = True):
        self.Pmfl = self.Sim_phot_mult(model_wave, model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)
        
        if scale == False:
            self.PC = 1
            
        self.Pmfl = self.Pmfl * self.PC
        
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self,  bfZ, bft, bftau, bfz, bfd):      
        self.sp.params['logzsol'] = np.log10(bfZ)
        self.sp.params['tau'] = bftau
        self.sp.params['dust2'] = bfd
        
        model_wave,model_flux = self.sp.get_spectrum(tage = bft, peraa = True)
        
        US_model_flux = F_lam_per_M(model_flux, model_wave * (1+bfz), bfz, 0, self.sp.stellar_mass)

        US_pfl = self.Sim_phot_mult(model_wave * (1+bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        if self.g102:  
            Bmf= self.Forward_model_all_beams(self.Bbeam, self.Bwv, model_wave * (1+bfz), 
                                                        self.S_model_flux)       
            Bmfl = Bmf / self.Btrans
            self.Bscale = Scale_model(self.Bfl, self.Ber, Bmfl)
            self.Bfl = self.Bfl / self.Bscale ; self.Ber = self.Ber / self.Bscale 
        
        if self.g141:    
            Rmf= self.Forward_model_all_beams(self.Rbeam, self.Rwv, model_wave * (1+bfz), 
                                                        self.S_model_flux) 
            Rmfl = Rmf / self.Rtrans
            self.Rscale = Scale_model(self.Rfl, self.Rer, Rmfl)
            self.Rfl = self.Rfl / self.Rscale ; self.Rer = self.Rer / self.Rscale 
        
        self.set_scale = True
        
    def Make_sim(self, model_wave, model_flux, specz, rndstate = 10, perturb = True):       
        self.SBfl, self.SBer,  self.SRfl, self.SRer, self.SPflx, self.SPerr =  init_sim(model_wave, 
            model_flux, specz, self.Bwv, self.Rwv, 
            self.Bfl, self.Rfl, self.Pflx, self.Ber, self.Rer, self.Perr, 0, 
            self.Btrans, self.Rtrans, self.Bbeam, self.Rbeam, 
            self.IDP, self.sens_wv, self.b, self.dnu, self.adj , rndstate = rndstate, perturb = perturb)
        
    def Forward_model_all_beams(self, beams, in_wv, model_wave, model_flux):
        return forward_model_all_beams(beams, in_wv, model_wave, model_flux)
    
    def Forward_model_all_beams_flatted(self, beams, trans, in_wv, model_wave, model_flux):
        return forward_model_all_beams_flatted(beams, trans, in_wv, model_wave, model_flux)
    
    def Sim_all_premade(self, model_wave, model_flux, scale = True):
        self.Sim_phot_premade(model_wave, model_flux, scale = scale)
        self.Sim_spec_premade(model_wave, model_flux)

###########
###########

class Gen_SF_spec(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                phot_errterm = 0, irac_err = None, mask = True, setup = False):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.g102_beam = glob(beam_path + '*{0}*g102*'.format(galaxy_id))
        self.g141_beam = glob(beam_path + '*{0}*g141*'.format(galaxy_id))
        self.sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau = 0.1, dust_type = 1)
        self.mask = mask
        
        if len(self.g102_beam) < 1 :
            self.g102_beam = glob(beam_path + '*{0}*g102*'.format(10639))
        if len(self.g141_beam) < 1 :
            self.g141_beam = glob(beam_path + '*{0}*g141*'.format(10639))
        
        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        """

        ##load spec and phot
        try:
            self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = load_spec_SF(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz, mask = self.mask, setup = setup)

            self.Bfl = self.Bflx / self.Bflt 
            self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, self.field, self.galaxy_id, 'G102')

            self.Ber = self.Berr / self.Bflt
            self.g102 = True
            if self.mask == False:
                self.Bmask = get_mask(self.field, self.galaxy_id, self.Bwv, 'g102')

        except:
            print('missing g102')
            self.g102 = False

        try:
            self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = load_spec_SF(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz, mask = self.mask, setup = setup)

            self.Rfl = self.Rflx / self.Rflt 
            self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, self.field, self.galaxy_id, 'G141')
            self.Rer = self.Rerr / self.Rflt
            self.g141 = True
            if self.mask == False:
                self.Rmask = get_mask(self.field, self.galaxy_id, self.Rwv, 'g141')

        except:
            print('missing g141')
            self.g141 = False
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec_SF(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False)
         
        self.Perr = apply_phot_err(self.Pflx, self.Perr, self.Pnum, base_err = phot_errterm, irac_err = irac_err)
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
    
    def Sim_spec_premade(self, model_wave, model_flux):
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam, self.Btrans, self.Bwv, model_wave, model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam, self.Rtrans, self.Rwv, model_wave, model_flux) 
            self.Rmfl *= self.PC

            if not self.set_scale:
                Rscale = Scale_model(self.Rfl, self.Rer, self.Rmfl)

                self.Rfl = self.Rfl / Rscale ; self.Rer = self.Rer / Rscale 
    
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)

    def Sim_phot_premade(self, model_wave, model_flux, scale = True):
        self.Pmfl = self.Sim_phot_mult(model_wave, model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)
        
        if scale == False:
            self.PC = 1
            
        self.Pmfl = self.Pmfl * self.PC
    
    @staticmethod
    def Forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
        return forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux)
    
    def Sim_all_premade(self, model_wave, model_flux, scale = True):
        self.Sim_phot_premade(model_wave, model_flux, scale = scale)
        self.Sim_spec_premade(model_wave, model_flux)
        
    def Best_fit_scale(self, model_wave, model_flux, specz, bp1, rp1, massperc,logmass):
        self.Full_forward_model(model_wave, F_lam_per_M(model_flux, model_wave * (1 + specz),
                            specz, 0, massperc)*10**logmass, specz)

        if self.g102:
            self.bcal = Calibrate_grism([self.Bwv, self.Bfl, self.Ber], self.Bmfl, bp1)
            if self.mask == False:
                self.bscale = Scale_model(self.Bfl[self.Bmask] / self.bcal[self.Bmask], 
                                          self.Ber[self.Bmask]/ self.bcal[self.Bmask], self.Bmfl[self.Bmask])

            else:
                self.bscale = Scale_model(self.Bfl / self.bcal, self.Ber/ self.bcal, self.Bmfl)
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
            
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)
            if self.mask == False:
                self.rscale = Scale_model(self.Rfl[self.Rmask] / self.rcal[self.Rmask], 
                                          self.Rer[self.Rmask]/ self.rcal[self.Rmask], self.Rmfl[self.Rmask])
                
            else:
                self.rscale = Scale_model(self.Rfl / self.rcal, self.Rer/ self.rcal, self.Rmfl)
            self.Rfl =  self.Rfl/ self.rcal/ self.rscale
            self.Rer =  self.Rer/ self.rcal/ self.rscale
            
    def Full_forward_model(self, model_wave, model_flux, specz):
        
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams(self.Bbeam , self.Btrans, self.Bwv, model_wave * (1 + specz), model_flux)

        if self.g141:
            self.Rmfl = self.Forward_model_all_beams(self.Rbeam , self.Rtrans, self.Rwv, model_wave * (1 + specz), model_flux)
            
        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + specz), model_flux)
        
    def Best_fit_scale_flam(self, model_wave, model_flam, specz, bp1, rp1):
        self.Full_forward_model(model_wave, model_flam, specz)

        if self.g102:
            self.bcal = Calibrate_grism([self.Bwv, self.Bfl, self.Ber], self.Bmfl, bp1)
            if self.mask == False:
                self.bscale = Scale_model(self.Bfl[self.Bmask] / self.bcal[self.Bmask], 
                                          self.Ber[self.Bmask]/ self.bcal[self.Bmask], self.Bmfl[self.Bmask])

            else:
                self.bscale = Scale_model(self.Bfl / self.bcal, self.Ber/ self.bcal, self.Bmfl)            
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
            
            
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)
            if self.mask == False:
                self.rscale = Scale_model(self.Rfl[self.Rmask] / self.rcal[self.Rmask], 
                                          self.Rer[self.Rmask]/ self.rcal[self.Rmask], self.Rmfl[self.Rmask])
                
            else:
                self.rscale = Scale_model(self.Rfl / self.rcal, self.Rer/ self.rcal, self.Rmfl)
            self.Rfl =  self.Rfl/ self.rcal/ self.rscale
            self.Rer =  self.Rer/ self.rcal/ self.rscale
    
    def Make_sim(self, model_wave, model_flux, specz, rndstate = 10, perturb = True):       
        self.SBfl, self.SBer,  self.SRfl, self.SRer, self.SPflx, self.SPerr =  init_sim(model_wave, 
            model_flux, specz, self.Bwv, self.Rwv, 
            self.Bfl, self.Rfl, self.Pflx, self.Ber, self.Rer, self.Perr, 0, 
            self.Btrans, self.Rtrans, self.Bbeam, self.Rbeam, 
            self.IDP, self.sens_wv, self.b, self.dnu, self.adj , rndstate = rndstate, perturb = perturb)
        
def Calibrate_grism(spec, Gmfl, p1):
    lines = (p1 * (spec[0] -(spec[0][-1] + spec[0][0])/2 ) + 1E3)
    scale = Scale_model(spec[1]  / lines, spec[2] / lines, Gmfl)    
    return scale * lines

##################################

class Gen_spec_2D(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8000, 11300], g141_lims=[11300, 16500],
                phot_errterm = 0, irac_err = None):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims

        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD
        galaxy_id - ID number from 3D-HST
        specz - redshift

        """
        ##load spec and phot
        self.Clean_multibeam()
        
        if self.g102:
            self.Bwv, self.Bfl, self.Ber = self.Gen_1D_spec(self.mb_g102, g102_lims, 'G102')
            self.Bwv_rf = self.Bwv/(1 + self.specz)
        
        if self.g141:
            self.Rwv, self.Rfl, self.Rer = self.Gen_1D_spec(self.mb_g141, g141_lims, 'G141')
            self.Rwv_rf = self.Rwv/(1 + self.specz)
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False, select = None)
         
        self.Perr = apply_phot_err(self.Pflx, self.Perr, self.Pnum, base_err = phot_errterm, irac_err = irac_err)
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
       
    def Sim_phot_premade(self, model_wave, model_flux):
        self.Pmfl = self.Sim_phot_mult(model_wave, model_flux)
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)

        self.Pmfl = self.Pmfl * self.PC
     
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)
    
    def Clean_multibeam(self):
        BMX = np.load(beam_path +'beam_config/{}_{}_ex.npy'.format(self.field, self.galaxy_id))
        clip, clipspec, omitspec = np.load(beam_path +'beam_config/{}_{}.npy'.format(self.field, self.galaxy_id))
        fl = glob(beam_2d_path + '*{}*/*{}*'.format(self.field[1], self.galaxy_id))

        
        sz = []
        for f in fl:
            sz.append(os.path.getsize(f))
       
        fl = np.array(fl)[np.argsort(sz)]

        nlist = []
        blist = []
        for f in fl:
            mb = multifit.MultiBeam(f,**args)
            for bm in mb.beams:
                if bm.grism.parent_file not in nlist:
                    nlist.append(bm.grism.parent_file)
                    blist.append(bm)

        #####clip or omit
        fblist = []

        idc = 0

        for bm in blist:
            if bm.grism.parent_file in BMX:            
                if clipspec[idc] == 1:
                    xspec, yspec, yerr = bm.beam.optimal_extract(bm.grism.data['SCI'] - bm.contam,ivar = bm.ivar) 
                    for lms in clip[idc]:
                        for i in range(len(xspec)):
                            if lms[0] < xspec[i]< lms[1]:
                                bm.grism.data['SCI'].T[i] = np.zeros_like(bm.grism.data['SCI'].T[i])
                                bm.grism.data['ERR'].T[i] = np.ones_like(bm.grism.data['ERR'].T[i])*1000  

                if omitspec[idc] == 1:
                    pass
                else:    
                    fblist.append(bm)

                idc += 1
    
            else:    
                fblist.append(bm)   


        mb = multifit.MultiBeam(fblist,**args)
        for b in mb.beams:
            if hasattr(b, 'xp'):
                delattr(b, 'xp')
        mb.initialize_masked_arrays()

        grism_beams = {}
        for g in mb.PA:
            grism_beams[g.lower()] = []
            for pa in mb.PA[g]:
                for i in mb.PA[g][pa]:
                    grism_beams[g.lower()].append(mb.beams[i])

        try:
            self.mb_g102 = multifit.MultiBeam(grism_beams['g102'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g102')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g102.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g102.initialize_masked_arrays()
            self.g102 = True
            
        except:
            self.g102 = False
            
        try:
            self.mb_g141 = multifit.MultiBeam(grism_beams['g141'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g141')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g141.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g141.initialize_masked_arrays()
            self.g141 = True
            
        except:
            self.g141 = False
            
    def Gen_1D_spec(self, MB, lims, instr, tfit = 'none'):
        if tfit != 'none':
            sptbl = MB.oned_spectrum(tfit = tfit)
        else:
            sptbl = MB.oned_spectrum()

        w = sptbl[instr]['wave']
        f = sptbl[instr]['flux']
        e = sptbl[instr]['err']
        fl = sptbl[instr]['flat']

        clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1]]
        
        return w[clip], f[clip]/fl[clip], e[clip]/fl[clip]