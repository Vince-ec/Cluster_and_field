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

def:
Best_fitter
Best_fitter_sim
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



def Best_fitter(field, galaxy, g102_beam, g141_beam, specz,
                errterm = 0):
    ######## initialize spec
    
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, specz, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm,)    
    
    metal_i = 0.019
    age_i = 2
    tau_i = 0.1
    rshift_i = specz
    dust_i = 0.1
    
    if Gs.g102:
        Bmwv,Bmflx = forward_model_grism(Gs.Bbeam, wave, flux)
    
    if Gs.g141:
        Rmwv,Rmflx = forward_model_grism(Gs.Rbeam, wave, flux)
    
    Pmflx = []

    instr = ['P','B','R']
    
    for u in instr:
        if u == 'P':
            mflx = Pmflx
            W = Gs.Pwv; F = Gs.Pflx; E = Gs.Perr; MW = Gs.Pwv; phot = True
        if u == 'B' and Gs.g102:
            mflx = Bmflx
            W = Gs.Bwv; F = Gs.Bflx; E = Gs.Berr; MW = Bmwv; phot = False
        if u == 'R' and Gs.g141:
            mflx = Rmflx
            W = Gs.Rwv; F = Gs.Rflx; E = Gs.Rerr; MW = Rmwv; phot = False
        
        try:  
            for x in range(3):

                metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)

                mfl = Gen_grid(Gs, sp, metal, age, tau, rshift, u, mflx)

                ## set some variables

                grid = Stitch_resize_redden_fit(W, F, E, mfl, MW, 
                                 metal, age, tau, rshift, dust, phot = phot) 

                bfd, bfZ, bft, bftau, bfz = Best_fit_model(grid, metal, age, tau, rshift, dust)

                metal_i = bfZ
                age_i = bft
                tau_i = bftau
                rshift_i = bfz
                dust_i = bfd

                print(u, bfZ, bft, bftau, bfz, bfd)   

        except:
            print('data missing')
            
        rshift_i = specz

def Best_fitter_sim(field, galaxy, g102_beam, g141_beam, specz, 
                    simZ, simt, simtau, simz, simd,
                    errterm = 0):
    ######## initialize spec
    
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, specz, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm,)    
    
    Gs.Make_sim(simZ, simt, simtau, simz, simd)
    
    metal_i = 0.019
    age_i = 2
    tau_i = 0.1
    rshift_i = simz
    dust_i = 0.1
    
    [Bmwv,Bmflx], [Rmwv,Rmflx] = Gs.Sim_spec_mult(wave, flux)

    
    for x in range(3):
    
        metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)
    
        Pmfl = Gen_Pgrid(Gs, sp, metal, age, tau, rshift)

        ## set some variables

        Pgrid = Stitch_resize_redden_fit(Gs.Pwv, Gs.SPflx, Gs.SPerr, Pmfl, Gs.Pwv, 
                         metal, age, tau, rshift, dust, phot=True) 
        
        bfd, bfZ, bft, bftau, bfz = Best_fit_model(Pgrid, metal, age, tau, rshift, dust)
        
        metal_i = bfZ
        age_i = bft
        tau_i = bftau
        rshift_i = bfz
        dust_i = bfd
        
        print('PHOT:', bfZ, bft, bftau, bfz, bfd)   
       
    rshift_i = simz

    for x in range(3):
    
        metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)
    
        Bmfl = Gen_Bgrid(Gs, sp, metal, age, tau, rshift)

        ## set some variables
        Bgrid = Stitch_resize_redden_fit(Gs.Bwv, Gs.SBflx, Gs.SBerr, Bmfl, Bmwv, 
                         metal, age, tau, rshift, dust)
        
        bfd, bfZ, bft, bftau, bfz = Best_fit_model(Bgrid, metal, age, tau, rshift, dust)
        
        metal_i = bfZ
        age_i = bft
        tau_i = bftau
        rshift_i = bfz
        dust_i = bfd
        
        print('G102:', bfZ, bft, bftau, bfz, bfd)   

    rshift_i = simz

    for x in range(3):
    
        metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)
    
        Rmfl = Gen_Rgrid(Gs, sp, metal, age, tau, rshift)

        ## set some variables
        Rgrid = Stitch_resize_redden_fit(Gs.Rwv, Gs.SRflx, Gs.SRerr, Rmfl, Rmwv, 
                         metal, age, tau, rshift, dust)

        bfd, bfZ, bft, bftau, bfz = Best_fit_model(Rgrid, metal, age, tau, rshift, dust)
        
        metal_i = bfZ
        age_i = bft
        tau_i = bftau
        rshift_i = bfz
        dust_i = bfd
        
        print('G141:', bfZ, bft, bftau, bfz, bfd)


    rshift_i = simz
 
    for x in range(3):
    
        metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)
    
        Pmfl = Gen_Pgrid(Gs, sp, metal, age, tau, rshift)
        Bmfl = Gen_Bgrid(Gs, sp, metal, age, tau, rshift)
        Rmfl = Gen_Rgrid(Gs, sp, metal, age, tau, rshift)

        ## set some variables
        Pgrid = Stitch_resize_redden_fit(Gs.Pwv, Gs.SPflx, Gs.SPerr, Pmfl, Gs.Pwv, 
                         metal, age, tau, rshift, dust, phot=True) 
        Bgrid = Stitch_resize_redden_fit(Gs.Bwv, Gs.SBflx, Gs.SBerr, Bmfl, Bmwv, 
                         metal, age, tau, rshift, dust)    
        Rgrid = Stitch_resize_redden_fit(Gs.Rwv, Gs.SRflx, Gs.SRerr, Rmfl, Rmwv, 
                         metal, age, tau, rshift, dust)

        bfd, bfZ, bft, bftau, bfz = Best_fit_model(Pgrid + Bgrid + Rgrid, metal, age, tau, rshift, dust)
        
        metal_i = bfZ
        age_i = bft
        tau_i = bftau
        rshift_i = bfz
        dust_i = bfd
        
        print('ALL:', bfZ, bft, bftau, bfz, bfd)


def Best_fit_model(chi, metal, age, tau, rshift, dust):
    x = np.argwhere(chi == np.min(chi))[0]
    return dust[x[0]],metal[x[1]], age[x[2]], tau[x[3]] , rshift[x[4]]
        
def Stitch_resize_redden_fit(fit_wv, fit_fl, fit_er, mfl, mwv, 
                     metal, age, tau, rshift, dust, phot=False):
    #############Read in spectra and stich spectra grid together#################
    if phot:
        chigrid = Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust)  
        return chigrid
    else:
        mfl = Resize(fit_wv, mwv, mfl)
        chigrid = Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust)  
        return chigrid
        
def Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, Av):    
    minidust = Gen_dust_minigrid(fit_wv, redshift, Av)

    chigrids = []
    
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
        redflgrid = mfl * dustgrid        
        SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
        redflgrid = np.array([SCL]).T*redflgrid
        chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
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


def Gen_grid(spec, models, metal, age, tau, rshift, instr, grism_flux = None):    
    ##### set model wave
    if instr == 'P'
        mfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(spec.IDP)])
    
        for i in range(len(metal)):
            models.params['logzsol'] = np.log10(metal[i] / 0.019)
            for ii in range(len(age)):
                for iii in range(len(tau)):
                    models.params['tau'] = tau[iii]
                    wv,fl = models.get_spectrum(tage = age[ii], peraa = True)
                    for iv in range(len(rshift)):
                        mfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = spec.Sim_phot_mult(wv * (1 + rshift[iv]),fl)
    
    if instr == 'B'
        mfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(grism_flux)])
        beam = spec.Bbeam
    
    if instr == 'R'
        mfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(grism_flux)])
        beam = spec.Rbeam      
      
    if instr != 'P'
        for i in range(len(metal)):
            models.params['logzsol'] = np.log10(metal[i] / 0.019)
            for ii in range(len(age)):
                for iii in range(len(tau)):
                    models.params['tau'] = tau[iii]
                    wv,fl = models.get_spectrum(tage = age[ii], peraa = True)
                    for iv in range(len(rshift)):
                        mwv,mflx= forward_model_grism(beam, wv * (1 + rshift[iv]),fl)
                        mfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = mflx
    return mfl

    
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

def Simple_analyze(chi, metal, age, tau, rshift, dust):
    ######## get Pd and Pz 
    P_full = np.exp(- chi / 2).astype(np.float128)

    P_full /= np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), tau, axis=3), age, axis=2), metal, axis=1),dust)
    
    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), tau, axis=3), age, axis=2), metal, axis=1) 

    Pz = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), tau, axis=1) 

    P = np.trapz(P_full, rshift, axis=4)
    P = np.trapz(P.T, dust, axis=3).T
   
    PZ = np.trapz(np.trapz(P, tau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, tau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return PZ, Pt, Ptau, Pz, Pd