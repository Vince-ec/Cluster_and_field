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
    Stitch_resize_redden_fit3, Stitch_spec, Scale_model_mult, Resize

from sim_engine import *

hpath = os.environ['HOME'] + '/'

"""
class:

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

class Gen_sim(object):
    def __init__(self, field, galaxy_id, specz, g102_beam, g141_beam,
                 model_wv, model_fl, mass, stellar_mass,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                tmp_err = True, phot_errterm = 0,offset_limit = 0.1):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.set_scale = False
        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        tmp_err - (flag) whether or not we apply a template error function (not available)
        """
        
        # load spec and phot
        self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB = load_spec(self.field,
                                self.galaxy_id, 'g102', self.g102_lims,  self.specz)
        
        self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR = load_spec(self.field,
                                self.galaxy_id, 'g141', self.g141_lims,  self.specz)
        
        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False)
        
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
        
        ### set beams
        self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, g102_beam)
        self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, g141_beam)
        
        ### set sim and transmission curve
        self.SBflx, self.SBer, self.SRflx, self.SRer, self.SPflx, self.SPer =  init_sim(model_wv, model_fl, 
            specz, mass, stellar_mass, self.Bwv, self.Rwv, self.Bflx, self.Rflx, self.Pflx, self.Berr, 
            self.Rerr, self.Perr, phot_errterm, self.Btrans, self.Rtrans, self.Bbeam, self.Rbeam, 
            self.IDP, self.sens_wv, self.b, self.dnu, self.adj, offset_limit)
            
        self.SBer = apply_tmp_err(self.Bwv_rf,self.SBer,self.SBflx, tmp_err = tmp_err)
        self.SRer = apply_tmp_err(self.Rwv_rf,self.SRer,self.SRflx, tmp_err = tmp_err)
        self.SPer = apply_tmp_err(self.Pwv_rf,self.SPer,self.SPflx, tmp_err = tmp_err, pht_err = phot_errterm)
        
    def Sim_spec_mult(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return forward_model_grism(self.Bbeam, model_wave, model_flux), \
                forward_model_grism(self.Rbeam, model_wave, model_flux)

    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))

        [Bmw, Bmf], [Rmw, Rmf] = self.Sim_spec_mult(model_wave * (1 + model_redshift), 
                                                    model_flux * Salmon(Av,model_wave))
        iBmf = interp1d(Bmw,Bmf)(self.Bwv)       
        iRmf = interp1d(Rmw,Rmf)(self.Rwv)     
        
        self.Bmfl = iBmf / self.Btrans
        self.Rmfl = iRmf / self.Rtrans
            
        self.Bmfl *= self.PC
        self.Rmfl *= self.PC
        
        if self.offset:
            Bscale = Scale_model(self.SBflx, self.SBer, self.Bmfl)
            Rscale = Scale_model(self.SRflx, self.SRer, self.Rmfl)

            self.SBflx = self.SBflx / Bscale ; self.SBer = self.SBer / Bscale 
            self.SRflx = self.SRflx / Rscale ; self.SRer = self.SRer / Rscale 
       
    def Sim_phot_mult(self, model_wave, model_flux):
        return forward_model_phot(model_wave, model_flux, self.IDP, self.sens_wv, self.b, self.dnu, self.adj)

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))
        
        self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), 
                                                  model_flux * Salmon(Av,model_wave))
        self.PC =  Scale_model(self.SPflx, self.SPer, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self, bfZ, bft, bftau, bfz, bfd):
        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(bfZ, bft, bftau))
        
        US_model_flux = F_lam_per_M(model_flux * Salmon(bfd,model_wave), model_wave * (1 + bfz), bfz, 0, 1)

        US_pfl = self.Sim_phot_mult(model_wave * (1 + bfz), US_model_flux)
        
        self.mass = Scale_model(self.SPflx, self.SPer, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        Bw,Bf = forward_model_grism(self.Bbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        Rw,Rf = forward_model_grism(self.Rbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        
        iBmf = interp1d(Bw,Bf)(self.Bwv)       
        iRmf = interp1d(Rw,Rf)(self.Rwv)  
  
        Bmfl = iBmf / self.Btrans
        Rmfl = iRmf /self.Rtrans

        self.Bscale = Scale_model(self.SBflx, self.SBer, Bmfl)
        self.Rscale = Scale_model(self.SRflx, self.SRer, Rmfl)
        
        self.SBflx = self.SBflx / self.Bscale ; self.SBer = self.SBer / self.Bscale 
        self.SRflx = self.SRflx / self.Rscale ; self.SRer = self.SRer / self.Rscale 
        
        self.set_scale = True


def Fit_all_sim(field, galaxy, g102_beam, g141_beam, specz, metal, age, tau, dust, 
                simZ, simt, simtau, simz, simd, name, gen_models = True, 
                age_conv= data_path + 'light_weight_scaling_3.npy', errterm = 0,
           outname = 'none'):
   
    if outname == 'none':
        outname = name
    ######## initialize spec
    
    fsps_spec = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(simZ/0.019), sfh = 4, tau = simtau)
    wave, flux = fsps_spec.get_spectrum(tage = simt, peraa = True)
    
    sal = Salmon(simd, wave)
    
    sp = Gen_sim(field, galaxy, specz, g102_beam, g141_beam,
                   wave, flux * sal, 11, fsps_spec.stellar_mass, g102_lims=[8500,11300], 
                   tmp_err=True, phot_errterm = errterm, offset_limit=0.5 )    
    if gen_models:
        Gen_mflgrid(sp, name, metal, age, tau, specz)

    ## set some variables
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv,fl)
    
    Stitch_resize_redden_fit2(sp.Pwv, sp.SPflx, sp.SPer, 'none', 'phot', name, sp.Pwv, 
                     metal, age, tau, specz, outname, phot = True) 
    Stitch_resize_redden_fit2(sp.Bwv, sp.SBflx, sp.SBer, sp.Btrans, 'g102', name, Bmwv, 
                     metal, age, tau, specz, outname)
    Stitch_resize_redden_fit2(sp.Rwv, sp.SRflx, sp.SRer, sp.Rtrans, 'g141', name, Rmwv, 
                     metal, age, tau, specz, outname)

    P, PZ, Pt, Ptau, Pd = Analyze_full_fit(outname, metal, age, tau, specz,
                                              dust=dust,age_conv = age_conv)

    np.save(out_path + '{0}_tZ_sim_pos_fs'.format(outname),P)
    np.save(out_path + '{0}_Z_sim_pos_fs'.format(outname),[metal,PZ])
    np.save(out_path + '{0}_t_sim_pos_fs'.format(outname),[age,Pt])
    np.save(out_path + '{0}_tau_sim_pos_fs'.format(outname),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_d_sim_pos_fs'.format(outname),[dust,Pd])
    
    
    bfd, bfZ, bft, bftau = Get_best_fit(outname, metal, age, tau, specz)
    
    print(bfZ, bft, bftau, specz, bfd)
    
    sp.Scale_flux(bfZ, bft, bftau, specz, bfd)

    Stitch_resize_redden_fit3(sp.Pwv, sp.SPflx, sp.SPer, 'none', 'phot', name, sp.Pwv, 
                     metal, age, tau, specz, outname, phot = True)     
    Stitch_resize_redden_fit3(sp.Bwv, sp.SBflx, sp.SBer, sp.Btrans, 'g102', name, Bmwv, 
                     metal, age, tau, specz, outname)
    Stitch_resize_redden_fit3(sp.Rwv, sp.SRflx, sp.SRer, sp.Rtrans, 'g141', name, Rmwv, 
                     metal, age, tau, specz, outname)   
    
    P, PZ, Pt, Ptau, Pd = Analyze_full_fit(outname, metal, age, tau, specz,
                                              dust=dust,age_conv = age_conv)

    np.save(out_path + '{0}_tZ_sim_pos'.format(outname),P)
    np.save(out_path + '{0}_Z_sim_pos'.format(outname),[metal,PZ])
    np.save(out_path + '{0}_t_sim_pos'.format(outname),[age,Pt])
    np.save(out_path + '{0}_tau_sim_pos'.format(outname),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_d_sim_pos'.format(outname),[dust,Pd])
    
def Get_best_fit(outname, metal, age, tau, specz, dust = np.arange(0,1.1,0.1)):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)
    
    ####### Read in file   
    chi = np.zeros([len(dust),len(metal),len(age),len(tau)])
    instr = ['g102','g141','phot']
    
    for i in instr:
        chifiles = [chi_path + '{0}_d{1}_{2}_chidata.npy'.format(outname, U, i) for U in range(len(dust))]
        chi += Stich_grids(chifiles)
    
    chi[ : , : , len(age[age <= max_age]):] = 1E5

    return Best_fit_model(chi,metal,age,tau)
    

def Best_fit_model(chi, metal, age, tau, dust = np.arange(0,1.1,.1)):
    x = np.argwhere(chi == np.min(chi))[0]
    return dust[x[0]],metal[x[1]], age[x[2]], tau[x[3]]

def Analyze_full_fit(outname, metal, age, tau, specz, dust = np.arange(0,1.1,0.1), age_conv=data_path + 'light_weight_scaling_3.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)
    
    ####### Read in file   
    chi = np.zeros([len(dust),len(metal),len(age),len(tau)])
    instr = ['g102','g141','phot']
    
    for i in instr:
        chifiles = [chi_path + '{0}_d{1}_{2}_chidata.npy'.format(outname, U, i) for U in range(len(dust))]
        chi += Stich_grids(chifiles)
    
    chi[ : , : , len(age[age <= max_age]):] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))
    
    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    ######## get Pd and Pz 
    P_full = np.exp(-chi/2).astype(np.float128)

    Pd = np.trapz(np.trapz(np.trapz(P_full, ultau, axis=3), age, axis=2), metal, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(P_full, ultau, axis=3), age, axis=2), metal, axis=1),dust)

    P = np.trapz(P_full.T, dust, axis=3).T
    new_P = np.zeros(P.T.shape)

    ######## Reshape likelihood to get light weighted age instead of age when marginalized
    for i in range(len(tau)):
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii],P.T[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(0, overhead[i][ii]))
        new_P[i] = frame.T

    ####### Create normalize probablity marginalized over tau
    P = new_P.T

    # test_prob = np.trapz(test_P, ultau, axis=2)
    C = np.trapz(np.trapz(np.trapz(P, ultau, axis=2), age, axis=1), metal)

    P /= C

    prob = np.trapz(P, ultau, axis=2)
    
    # #### Get Z, t, tau, and z posteriors
    PZ = np.trapz(np.trapz(P, ultau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, ultau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return prob.T, PZ, Pt, Ptau, Pd

def Stitch_resize_redden_fit2(fit_wv, fit_fl, fit_er, fit_flat, instrument, name, mwv, 
                     metal, age, tau, specz, outname, phot=False):
    #############Read in spectra and stich spectra grid together#################
    files = [chi_path + 'spec_files/{0}_m{1}_{2}.npy'.format(name, U, instrument) for U in metal]
    mfl = Stitch_spec(files)
    
    if phot:
        Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname,phot = True)  
    
    else:
        mfl = Resize(fit_wv, fit_flat, mwv, mfl)
        Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname)  

def Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname, phot = False):    
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        sal = Salmon(Av[i], fit_wv/(1+specz))
        redflgrid = mfl * sal
        
        if phot:
            SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
            redflgrid = np.array([SCL]).T*redflgrid
            chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)])
            np.save(chi_path + '{0}_d{1}_{2}_SCL'.format(outname, i, instrument), np.array([SCL]).T)
        
        else:
            SCL = np.load(chi_path + '{0}_d{1}_phot_SCL.npy'.format(outname, i))
            redflgrid = SCL*redflgrid
            SCL2 = Scale_model_mult(fit_fl,fit_er,redflgrid)
            chigrid = np.sum(((fit_fl / np.array([SCL2]).T - redflgrid) / (fit_er / np.array([SCL2]).T)) ** 2, axis=1).reshape([len(metal), len(age), len(tau)])
        
        np.save(chi_path + '{0}_d{1}_{2}_chidata'.format(outname, i, instrument),chigrid)

def Gen_mflgrid(spec, name, metal, age, tau, specz):
    wv,fl = np.load(model_path + 'm0.019_a2.8_dt0_spec.npy')

    [Bmwv,Bmf_len], [Rmwv,Rmf_len] = spec.Sim_spec_mult(wv,fl)
    
    ##### set model wave
    for i in range(len(metal)):
        
        Bmfl = np.zeros([len(age)*len(tau),len(Bmf_len)])
        Rmfl = np.zeros([len(age)*len(tau),len(Rmf_len)])
        Pmfl = np.zeros([len(age)*len(tau),len(spec.IDP)])

        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(metal[i], age[ii], tau[iii]))
                [Bmwv,Bmflx], [Rmwv,Rmflx] = spec.Sim_spec_mult(wv * (1 + specz),fl)
                Pmflx = spec.Sim_phot_mult(wv * (1 + specz),fl)

                Bmfl[ii*len(tau) + iii] = Bmflx
                Rmfl[ii*len(tau) + iii] = Rmflx
                Pmfl[ii*len(tau) + iii] = Pmflx
        
        np.save(chi_path + 'spec_files/{0}_m{1}_g102'.format(name, metal[i]),Bmfl)
        np.save(chi_path + 'spec_files/{0}_m{1}_g141'.format(name, metal[i]),Rmfl)
        np.save(chi_path + 'spec_files/{0}_m{1}_phot'.format(name, metal[i]),Pmfl)
        
def Stitch_resize_redden_fit3(fit_wv, fit_fl, fit_er, fit_flat, instrument, name, mwv, 
                     metal, age, tau, specz, outname, phot=False):
    #############Read in spectra and stich spectra grid together#################
    files = [chi_path + 'spec_files/{0}_m{1}_{2}.npy'.format(name, U, instrument) for U in metal]
    mfl = Stitch_spec(files)
    
    if phot:
        Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname,phot = True)  
    
    else:
        mfl = Resize(fit_wv, fit_flat, mwv, mfl)
        Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname)  

def Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, specz, instrument, name, outname, phot = False):    
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        sal = Salmon(Av[i], fit_wv/(1+specz))
        redflgrid = mfl * sal
        
        if phot:
            SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
            redflgrid = np.array([SCL]).T*redflgrid
            chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)])
            np.save(chi_path + '{0}_d{1}_{2}_SCL'.format(outname, i, instrument), np.array([SCL]).T)
        
        else:
            SCL = np.load(chi_path + '{0}_d{1}_phot_SCL.npy'.format(outname, i))
            chigrid = np.sum(((fit_fl - redflgrid * SCL) / (fit_er)) ** 2, axis=1).reshape([len(metal), len(age), len(tau)])
        
        np.save(chi_path + '{0}_d{1}_{2}_chidata'.format(outname, i, instrument),chigrid)
