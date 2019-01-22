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
from spec_exam import Gen_spec
from grizli import multifit
from grizli import model
from astropy.cosmology import Planck13 as cosmo
import fsps
from C_full_fit import Scale_model_mult, Stitch_spec
from time import time
from sim_engine import *
from matplotlib import gridspec
hpath = os.environ['HOME'] + '/'

"""
class:

def:
Best_fitter
Best_fitter_sim
Redshift_fitter
Best_fit_model
Stich_resize_and_fit
Stich_grids
Fit
Resize
Gen_grid  
Set_params
Set_rshift_params
Simple_analyze
Redshift_analyze

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
    temp_out = '/tmp/'
    
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
    temp_out = '../data/tmp/'

def Fitter_sim(field, galaxy, g102_beam, g141_beam, specz, 
                    simZ, simt, simtau, simz, simd,
                mod = '',errterm = 0, decontam = False):
    ######## initialize spec
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, simz, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                phot_errterm = errterm, decontam = decontam)  
    
    print(simZ, simt, simtau, simz, simd)
    
    Gs.Make_sim(simZ, simt, simtau, simz, simd)
    
    # set dummy value
    metal_i = 1
    age_i = 1
    tau_i = 1
    rshift_i = simz
    dust_i = 1
    
    bfm = []
    bfa = []
    bft = []
    bfz = []
    bfd = []
    
    if Gs.g102:
        Bmwv,Bmflx = forward_model_grism(Gs.Bbeam, wave, flux)
        instr = np.array(['P','B'])

    if Gs.g141:
        Rmwv,Rmflx = forward_model_grism(Gs.Rbeam, wave, flux)
        instr = np.array(['P','R'])
 
    if Gs.g102 and Gs.g141:
        instr = np.array(['P','B','R'])

    Pmflx = []
    for x in range(5):
        grids = []
        
        for u in instr:
            if u == 'P':
                mflx = Pmflx
                W = Gs.Pwv; F = Gs.SPflx; E = Gs.SPerr; MW = Gs.Pwv; phot = True
            if u == 'B' and Gs.g102:
                mflx = Bmflx
                W = Gs.Bwv; F = Gs.SBflx; E = Gs.SBerr; MW = Bmwv; phot = False
            if u == 'R' and Gs.g141:
                mflx = Rmflx
                W = Gs.Rwv; F = Gs.SRflx; E = Gs.SRerr; MW = Rmwv; phot = False

            metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)

            Gen_grid(Gs, sp, metal, age, tau, rshift, dust, u, mflx)

            ## set some variables
            grids.append(Stich_resize_and_fit(W, F, E, MW, 
                             metal, age, tau, rshift, dust, phot = phot))
           
        
        Gen_lwagrid(sp, metal, age, tau, x, mod)
        
        mchi = np.min(np.array(sum(grids)))
        
        print(mchi)
        
        PZ, Pt, Ptau, Pz, Pd = Full_analyze(np.array(sum(grids)), mchi, metal, age, tau, rshift, dust, x, mod)
                
        np.save(temp_out + 'tmp_chi_{0}_v{1}'.format(x, mod), np.array(sum(grids)))

        metal_i = np.round(metal[PZ == max(PZ)],4)
        age_i = np.round(age[Pt == max(Pt)],4)
        tau_i = np.round(tau[Ptau == max(Ptau)],4)        
        rshift_i = np.round(rshift[Pz == max(Pz)],4)
        dust_i = np.round(dust[Pd == max(Pd)],4)

        print(metal_i)   
        print(age_i)       
        print(tau_i)       
        print(rshift_i)   
        print(dust_i)       
        
        bfm.append(metal_i)   
        bfa.append(age_i)    
        bft.append(tau_i)               
        bfz.append(rshift_i) 
        bfd.append(dust_i)  
        
    np.save(temp_out + 'tmp_best_fits_v{0}'.format(mod), [bfm,bfa,bft,bfz,bfd])
    np.save(temp_out + 'tmp_grism_combo_v{0}'.format(mod),[g102_beam, g141_beam])

    
def Best_fitter(field, galaxy, g102_beam, g141_beam, specz, 
                mod = '',errterm = 0, decontam = False):
    ######## initialize spec
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, specz, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm, decontam = decontam)  
    
    # set dummy value
    metal_i = 1
    age_i = 1
    tau_i = 1
    rshift_i = specz
    dust_i = 1
    
    bfm = []
    bfa = []
    bft = []
    bfz = []
    bfd = []
    
    if Gs.g102:
        Bmwv,Bmflx = forward_model_grism(Gs.Bbeam, wave, flux)
        instr = np.array(['P','B'])

    if Gs.g141:
        Rmwv,Rmflx = forward_model_grism(Gs.Rbeam, wave, flux)
        instr = np.array(['P','R'])
 
    if Gs.g102 and Gs.g141:
        instr = np.array(['P','B','R'])

    Pmflx = []
    for x in range(5):
        grids = []
        
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

            metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)

            Gen_grid(Gs, sp, metal, age, tau, rshift, dust, u, mflx)

            ## set some variables
            grids.append(Stich_resize_and_fit(W, F, E, MW, 
                             metal, age, tau, rshift, dust, phot = phot))
           
        mchi = np.min(np.array(sum(grids)))
        
        print(mchi)
        #PZ, Pt, Ptau, Pz, Pd =  Simple_analyze_full(np.array(sum(grids)), mchi, metal, age, tau, rshift, dust)
        
        metal_i, age_i, tau_i, rshift_i, dust_i = Best_fit_model(np.array(sum(grids)), metal, age, tau, rshift, dust)
        
        np.save(temp_out + 'tmp_chi_{0}_v{1}'.format(x, mod), np.array(sum(grids)))

        #metal_i = np.round(metal[PZ == max(PZ)],4)
        #age_i = np.round(age[Pt == max(Pt)],4)
        #tau_i = np.round(tau[Ptau == max(Ptau)],4)        
        #rshift_i = np.round(rshift[Pz == max(Pz)],4)
        #dust_i = np.round(dust[Pd == max(Pd)],4)

        print(metal_i)   
        print(age_i)       
        print(tau_i)       
        print(rshift_i)   
        print(dust_i)       
        
        bfm.append(metal_i)   
        bfa.append(age_i)    
        bft.append(tau_i)               
        bfz.append(rshift_i) 
        bfd.append(dust_i)  
        
    np.save(temp_out + 'tmp_best_fits_v{0}'.format(mod), [bfm,bfa,bft,bfz,bfd])
    np.save(temp_out + 'tmp_grism_combo_v{0}'.format(mod),[g102_beam, g141_beam])

def Get_bestfits(field, galaxy):
    versions = len(glob(temp_out + 'tmp_chi_*')) // 5

    BFM, BFA, BFT, BFZ, BFD = [0,0,0,0,0]
    G1,G2 = ['','']
    minchi = 1E5
    # rederive P(z)s
    for i in range(versions):
        g1,g2 = np.load(temp_out + 'tmp_grism_combo_v{0}.npy'.format(i))
        bfm,bfa,bft,bfz,bfd = np.load(temp_out + 'tmp_best_fits_v{0}.npy'.format(i))

        for ii in range(5):
            chi = np.load(temp_out + 'tmp_chi_{0}_v{1}.npy'.format(ii,i))
            if np.min(chi) < minchi:
                minchi = np.min(chi)
                G1,G2 = [g1,g2]
                BFM, BFA, BFT, BFZ, BFD = [bfm[ii], bfa[ii], bft[ii], bfz[ii], bfd[ii]]

    np.save(out_path + '{0}_{1}_bestfits'.format(field, galaxy),[ G1, G2, BFM, BFA, BFT, BFZ, BFD])

    #remove temporary files
    all_temps = glob(temp_out + 'tmp*npy')
    [os.remove(U) for U in all_temps]

def Best_fitter_sim(field, galaxy, g102_beam, g141_beam, specz, 
                    simZ, simt, simtau, simz, simd, mod = '',
                    errterm = 0, decontam = False):
    ######## initialize spec
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, specz, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm, decontam = decontam)  
    
    Gs.Make_sim(simZ, simt, simtau, simz, simd)
    
    # set dummy value
    metal_i = 1
    age_i = 1
    tau_i = 1
    rshift_i = simz
    dust_i = 1
    
    bfm = []
    bfa = []
    bft = []
    bfz = []
    bfd = []
    
    if Gs.g102:
        Bmwv,Bmflx = forward_model_grism(Gs.Bbeam, wave, flux)
        instr = np.array(['P','B'])

    if Gs.g141:
        Rmwv,Rmflx = forward_model_grism(Gs.Rbeam, wave, flux)
        instr = np.array(['P','R'])
 
    if Gs.g102 and Gs.g141:
        instr = np.array(['P','B','R'])

    Pmflx = []
    for x in range(5):
        grids = []
        
        for u in instr:
            if u == 'P':
                mflx = Pmflx
                W = Gs.Pwv; F = Gs.SPflx; E = Gs.SPerr; MW = Gs.Pwv; phot = True
            if u == 'B' and Gs.g102:
                mflx = Bmflx
                W = Gs.Bwv; F = Gs.SBflx; E = Gs.SBerr; MW = Bmwv; phot = False
            if u == 'R' and Gs.g141:
                mflx = Rmflx
                W = Gs.Rwv; F = Gs.SRflx; E = Gs.SRerr; MW = Rmwv; phot = False

            metal, age, tau, rshift, dust = Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)

            Gen_grid(Gs, sp, metal, age, tau, rshift, dust, u, mflx)

            ## set some variables
            grids.append(Stich_resize_and_fit(W, F, E, MW, 
                             metal, age, tau, rshift, dust, phot = phot))
           
        mchi = np.min(np.array(sum(grids)))
        
        print(mchi)
        PZ, Pt, Ptau, Pz, Pd =  Simple_analyze_full(np.array(sum(grids)), mchi, metal, age, tau, rshift, dust)
        
        np.save(temp_out + 'tmp_chi_{0}_v{1}'.format(x, mod), np.array(sum(grids)))

        metal_i = np.round(metal[PZ == max(PZ)],4)
        age_i = np.round(age[Pt == max(Pt)],4)
        tau_i = np.round(tau[Ptau == max(Ptau)],4)        
        rshift_i = np.round(rshift[Pz == max(Pz)],4)
        dust_i = np.round(dust[Pd == max(Pd)],4)

        print(metal_i)   
        print(age_i)       
        print(tau_i)       
        print(rshift_i)   
        print(dust_i)       
        
        bfm.append(metal_i)   
        bfa.append(age_i)    
        bft.append(tau_i)               
        bfz.append(rshift_i) 
        bfd.append(dust_i)  
        
    np.save(temp_out + 'tmp_best_fits_v{0}'.format(mod), [bfm,bfa,bft,bfz,bfd])

def Redshift_fitter(field, galaxy, g102_beam, g141_beam, mod = '',
                errterm = 0, decontam = True):
    ######## initialize spec
    
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.019/0.019), sfh = 4, tau = 0.1)
    wave, flux = sp.get_spectrum(tage = 2.0, peraa = True)
   
    Gs = Gen_spec(field, galaxy, 1, g102_beam, g141_beam,
                   g102_lims=[7000, 12000], g141_lims=[10000, 18000],
                   tmp_err=False, phot_errterm = errterm, decontam = decontam)    
    
    # set dummy value
    metal_i = 1
    age_i = 1
    tau_i = 1
    rshift_i = 1
    dust_i = 1
    
    bfm = []
    bfa = []
    bfz = []
    
    mchi = 0
    if Gs.g102:
        Bmwv,Bmflx = forward_model_grism(Gs.Bbeam, wave, flux)
        instr = np.array(['P','B'])

    if Gs.g141:
        Rmwv,Rmflx = forward_model_grism(Gs.Rbeam, wave, flux)
        instr = np.array(['P','R'])
 
    if Gs.g102 and Gs.g141:
        instr = np.array(['P','B','R'])

    Pmflx = []
    print(instr)
    for x in range(4):
        grids = []
        
        for u in instr:
#            try:  
            if u == 'P':
                mflx = Pmflx
                W = Gs.Pwv; F = Gs.Pflx; E = Gs.Perr; MW = Gs.Pwv; phot = True
            if u == 'B' and Gs.g102:
                mflx = Bmflx
                W = Gs.Bwv; F = Gs.Bflx; E = Gs.Berr; MW = Bmwv; phot = False
            if u == 'R' and Gs.g141:
                mflx = Rmflx
                W = Gs.Rwv; F = Gs.Rflx; E = Gs.Rerr; MW = Rmwv; phot = False

            metal, age, tau, rshift, dust = Set_rshift_params(metal_i, age_i, tau_i, rshift_i, dust_i, x)

            Gen_grid(Gs, sp, metal, age, tau, rshift, dust, u, mflx)

            ## set some variables
            grids.append(Stich_resize_and_fit(W, F, E, MW, 
                             metal, age, tau, rshift, dust, phot = phot))

#            except:
#                print('{0} data missing'.format(u))
           
        if mchi == 0:
            mchi = np.min(np.array(sum(grids)))
        
        PZ, Pt, Ptau, Pz, Pd =  Simple_analyze(np.array(sum(grids)), mchi, metal, age, tau, rshift, dust)
        
        np.save(temp_out + 'tmp_chi_{0}_v{1}'.format(x, mod), np.array(sum(grids)))
        
        metal_i = np.round(metal[PZ == max(PZ)],4)
        age_i = np.round(age[Pt == max(Pt)],4)
        rshift_i = np.round(rshift[Pz == max(Pz)],4)

        print(metal_i)   
        print(age_i)       
        print(rshift_i)   
        
        bfm.append(metal_i)   
        bfa.append(age_i)       
        bfz.append(rshift_i)  
        
    np.save(temp_out + 'tmp_best_fits_v{0}'.format(mod), [bfm,bfa,bfz])

def Redshift_get_Pz(field, galaxy):
    versions = len(glob(temp_out + 'tmp_chi_*')) // 4

    # get true minimum
    MIN = 1E18
    for i in range(versions):
        for ii in range(4):
            chi = np.load(temp_out + 'tmp_chi_{0}_v{1}.npy'.format(ii,i))
            if np.min(chi) < MIN:
                MIN = np.min(chi)

    # rederive P(z)s
    for i in range(versions):
        bfm,bfa,bfz = np.load(temp_out + 'tmp_best_fits_v{0}.npy'.format(i))
        bfm = np.append([1],bfm)
        bfa = np.append([1],bfa)
        bfz = np.append([1],bfz)
        
        for ii in range(4):
            mdist, adist, tdist, zdist, ddist = Set_rshift_params(bfm[ii], bfa[ii], 1, bfz[ii], 1, ii)
            chi = np.load(temp_out + 'tmp_chi_{0}_v{1}.npy'.format(ii,i))
            PZ, Pt, Ptau, Pz, Pd =  Simple_analyze(chi, MIN, mdist, adist, tdist, zdist, ddist)
            np.save(temp_out + 'tmp_fitz_{0}_v{1}'.format(ii,i), [zdist,Pz])

    hrz = np.arange(0,3.4+0.0001,0.0001)

    zgrid = np.zeros([4 * versions, len(hrz)])
    wgrid = np.zeros(zgrid.shape)

    for i in range(versions):
        for ii in range(4):
            z, Pz = np.load(temp_out + 'tmp_fitz_{0}_v{1}.npy'.format(ii,i))
            iPz = interp1d(z,Pz)
            for iii in range(len(hrz)):
                if z[0] <= hrz[iii] <= z[-1]:
                    zgrid[i*4 + ii][iii] = iPz(hrz[iii])
                    wgrid[i*4 + ii][iii] = 1                
    
    #stack posteriors
    pz_stack = np.zeros(len(hrz))
    for i in range(len(hrz)):
        pz_stack[i] = np.sum(zgrid.T[i] * wgrid.T[i]) / (np.sum(wgrid.T[i]))

    pz_stack /= np.trapz(pz_stack,hrz)

    #save result
    np.save(out_path + '{0}_{1}_Pofz'.format(field, galaxy), [hrz,pz_stack])
    
    #remove temporary files
    all_temps = glob(temp_out + 'tmp*npy')
    [os.remove(U) for U in all_temps]
    
def Best_fit_model(chi, metal, age, tau, rshift, dust):
    max_age = Oldest_galaxy(max(rshift))
    
    chi[ : , len(age[age <= max_age]):] = 1E15

    x = np.argwhere(chi == np.min(chi))[0]
    return metal[x[0]], age[x[1]], tau[x[2]] , rshift[x[3]], dust[x[4]]
        
def Stich_resize_and_fit(fit_wv, fit_fl, fit_er, mwv, 
                     metal, age, tau, rshift, dust, phot=False):
    #############Read in spectra and stich spectra grid together#################
    files = [temp_out + 'tmp_m{0}_spec.npy'.format(U) for U in range(len(metal))]
    mfl = Stitch_spec(files)
    
    if phot:
        chigrid = Fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust)  

    else:
        mfl = Resize(fit_wv, mwv, mfl)
        chigrid = Fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, rshift, dust)  
        
    return chigrid
  
def Stich_grids(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))
    return np.array(stc)
    
def Fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, dust):    
    chigrids = []
    mfl = mfl.reshape([len(metal), len(age)*len(tau)*len(redshift)*len(dust),len(fit_wv)])
    for i in range(len(metal)):
    
        SCL = Scale_model_mult(fit_fl,fit_er,mfl[i])
        mfl[i] = np.array([SCL]).T*mfl[i]
        chigrids.append( np.sum(((fit_fl - mfl[i]) / fit_er) ** 2, axis=1).reshape([len(age), len(tau), len(redshift), len(dust)]))
    return np.array(chigrids)


def Resize(fit_wv, mwv, mfl):
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(mwv,range(len(mfl.data)),mfl.data)(fit_wv,range(len(mfl.data)))
    return mfl


def Gen_grid(spec, models, metal, age, tau, rshift, dust, instr, grism_flux = None):    
    ### set dust grid:
    dust_grid = []
    
    wv, fl = models.get_spectrum(tage = 2.0, peraa = True)
    for i in dust:
        dust_grid.append(Salmon(i,wv))
        
    dust_grid = np.array(dust_grid)
    
    
    if instr == 'P':
        
        for i in range(len(metal)):            
            mfl = np.zeros([len(dust)*len(age)*len(tau)*len(rshift),len(spec.IDP)])
            models.params['logzsol'] = np.log10( metal[i] / 0.019)
            for ii in range(len(age)):
                for iii in range(len(tau)):
                    models.params['tau'] = tau[iii]
                    wv, fl = models.get_spectrum(tage = age[ii], peraa = True)
                    for iv in range(len(rshift)):
                        for v in range(len(dust)):
                            mfl[ii*len(tau)*len(rshift)*len(dust) + \
                                iii*len(rshift)*len(dust) + iv*len(dust) + v] = \
                            spec.Sim_phot_mult(wv * (1 + rshift[iv]),fl * dust_grid[v])
                            
            np.save(temp_out + 'tmp_m{0}_spec'.format(i), mfl)  
      
    if instr != 'P':
        for i in range(len(metal)):
            
            if instr == 'B':
                mfl = np.zeros([len(dust)*len(age)*len(tau)*len(rshift),len(grism_flux)])
                beam = spec.Bbeam
    
            if instr == 'R':
                mfl = np.zeros([len(dust)*len(age)*len(tau)*len(rshift),len(grism_flux)])
                beam = spec.Rbeam    
            
            models.params['logzsol'] = np.log10( metal[i] / 0.019)
            for ii in range(len(age)):
                for iii in range(len(tau)):
                    models.params['tau'] = tau[iii]
                    wv, fl = models.get_spectrum(tage = age[ii], peraa = True)
                    for iv in range(len(rshift)):
                        for v in range(len(dust)):
                            mwv,mflx= forward_model_grism(beam, wv * (1 + rshift[iv]), fl* dust_grid[v])
                            mfl[ii*len(tau)*len(rshift)*len(dust) + \
                                iii*len(rshift)*len(dust) + iv*len(dust) + v] =mflx
                            
            np.save(temp_out + 'tmp_m{0}_spec'.format(i), mfl)

def Gen_lwagrid(models, metal, age, tau, v, mod):
    models.params['compute_light_ages'] = True  
    
    lwa_grid = np.zeros([metal.size,age.size,tau.size])
    
    for i in range(len(metal)):
        models.params['logzsol'] = np.log10(metal[i]/0.019)
        for ii in range(len(age)):
            for iii in range(len(tau)):
                models.params['tau'] = tau[iii]
                wv_lwa, lwa_spec=np.array(models.get_spectrum(tage=age[ii]))
                lwa_grid[i][ii][iii] = interp1d(wv_lwa,lwa_spec)(4750)

    models.params['compute_light_ages'] = False  

    np.save(temp_out + 'tmp_lwa_scale_{0}_v{1}'.format(v, mod),lwa_grid)
            
def Set_params(metal_i, age_i, tau_i, rshift_i, dust_i, stage):
    if stage == 0:
        age = np.round(np.arange(0.1, 8.2, 0.5),2)
        metal= np.round(np.arange(0.001 , 0.032, 0.0045),4)
        tau = np.round(np.arange(0.25,2.251,0.25), 3)
        rshift = np.round(np.arange( rshift_i - 0.008, rshift_i + 0.0081, 0.002),4)
        dust = np.round(np.arange(0, 1.9, 0.3),2)
    
    if stage == 1:
        if age_i <=2:
            age_i = 2.1
        if metal_i <= 0.009:
            metal_i = 0.01    
        if tau_i <= 0.4:
            tau_i = 0.41 
        if dust_i <= 0.6:
            dust_i = 0.6  
            
        age = np.round(np.arange(age_i  - 2., age_i  + 2.1, 0.25),2)
        metal= np.round(np.arange(metal_i - 0.009, metal_i + 0.0091, 0.003),4)
        tau = np.round(np.arange(tau_i - 0.4,tau_i + 0.41,0.1), 3)
        rshift = np.round(np.arange( rshift_i - 0.004, rshift_i + 0.0041, 0.001),4)
        dust = np.round(np.arange(dust_i - 0.6, dust_i + 0.61, 0.2),2)
    
    if stage == 2:
        if age_i <= 0.8:
            age_i = 0.9
        if metal_i <= 0.006:
            metal_i = 0.007
        if tau_i <= 0.2:
            tau_i = 0.21   
        if dust_i <= 0.3:
            dust_i = 0.3 
            
        age = np.round(np.arange(age_i  - 0.8, age_i  + 0.81, 0.1),2)
        metal= np.round(np.arange(metal_i - 0.006, metal_i + 0.0061, 0.002),4)
        tau = np.round(np.arange(tau_i - 0.2,tau_i + 0.21,0.05), 3)
        rshift = np.round(np.arange( rshift_i - 0.002, rshift_i + 0.0021, 0.0005),4)
        dust = np.round(np.arange(dust_i - 0.3, dust_i + 0.31, 0.1),3)
    
    if stage == 3:
        if age_i <= 0.4:
            age_i = 0.5    
        if metal_i <= 0.003:
            metal_i = 0.004              
        if tau_i <= 0.1:
            tau_i = 0.11      
        if dust_i <= 0.15:
            dust_i = 0.15 
            
        age = np.round(np.arange(age_i  - 0.4, age_i  + 0.41, 0.05),2)
        metal= np.round(np.arange(metal_i - 0.003, metal_i + 0.0031, 0.001),4)
        tau = np.round(np.arange(tau_i - 0.1,tau_i + 0.11,0.025), 3)
        rshift = np.round(np.arange( rshift_i - 0.0008, rshift_i + 0.00081, 0.0002),4)
        dust = np.round(np.arange(dust_i - 0.15, dust_i + 0.16, 0.05),3)
    
    if stage == 4:
        if age_i <= 0.2:
            age_i = 0.3    
        if metal_i <= 0.0015:
            metal_i = 0.0025              
        if tau_i <= 0.04:
            tau_i = 0.05      
        if dust_i <= 0.06:
            dust_i = 0.06 
            
        age = np.round(np.arange(age_i  - 0.2, age_i  + 0.21, 0.025),2)
        metal= np.round(np.arange(metal_i - 0.0015, metal_i + 0.0016, 0.0005),4)
        tau = np.round(np.arange(tau_i - 0.04,tau_i + 0.041,0.01), 3)
        rshift = np.round(np.arange( rshift_i - 0.0004, rshift_i + 0.00041, 0.0001),4)
        dust = np.round(np.arange(dust_i - 0.06, dust_i + 0.07, 0.02),3)
    
    
    return metal, age, tau, rshift, dust

def Set_rshift_params(metal_i, age_i, tau_i, rshift_i, dust_i, stage):
    tau = np.round(np.logspace(np.log10(0.01), np.log10(3), 5), 3)
    dust = np.round(np.arange(0, 1.1, 0.3),2)

    
    if stage == 0:
        age = np.round(np.arange(0.5, 14.1, 2),2)
        metal= np.round(np.arange(0.002 , 0.031, 0.0045),4)
        rshift = np.arange(0, 3.5, 0.1)
    
    if stage == 1:
        if age_i <= 3:
            age_i = 3.1
            
        if metal_i <= 0.006:
            metal_i = 0.007
            
        age = np.round(np.arange(age_i - 3, age_i + 4, 1),2)
        metal= np.round(np.arange(metal_i - 0.006, metal_i + 0.007, 0.002),4)
        rshift = np.arange(rshift_i - 0.17, rshift_i + 0.18, 0.01)
        
    
    if stage == 2:
        if age_i <= 1.5:
            age_i = 1.6
            
        if metal_i <= 0.003:
            metal_i = 0.004
            
        age = np.round(np.arange(age_i - 1.5, age_i + 2, 0.5),2)
        metal= np.round(np.arange(metal_i - 0.003, metal_i + 0.004, 0.001),4)
        rshift = np.arange(rshift_i - 0.017, rshift_i + 0.0171, 0.001)
   
    if stage == 3:
        if age_i <= 0.3:
            age_i = 0.4
            
        if metal_i <= 0.0015:
            metal_i = 0.0025
            
        age = np.round(np.arange(age_i - 0.3, age_i + 0.4, 0.1),2)
        metal= np.round(np.arange(metal_i - 0.0015, metal_i + 0.002, 0.0005),4)
        rshift = np.arange(rshift_i - 0.0017, rshift_i + 0.00171, 0.0001)
    
    return metal, age, tau, rshift, dust


def Simple_analyze_full(chi, mchi, metal, age, tau, rshift, dust):
        ####### Get maximum age
    max_age = Oldest_galaxy(max(rshift))
    
    chi[ : , len(age[age <= max_age]):] = 1E15
    
    ######## get Pd and Pz 
    P_full = np.exp(- (chi - mchi) / 2).astype(np.float128)
    
    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, metal, axis=4), age, axis=3), tau, axis=2), rshift, axis=1) 

    P = np.trapz(P_full, dust, axis=4)
    
    Pz = np.trapz(np.trapz(np.trapz(P.T, metal, axis=3), age, axis=2), tau, axis=1) 

    P = np.trapz(P, rshift, axis=3)
   
    PZ = np.trapz(np.trapz(P, tau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, tau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return PZ, Pt, Ptau, Pz, Pd

def Simple_analyze(chi, mchi, metal, age, tau, rshift, dust):
    ######## get Pd and Pz 
    P_full = np.exp(- (chi - mchi) / 2).astype(np.float128)
    
    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, metal, axis=4), age, axis=3), tau, axis=2), rshift, axis=1) 

    P = np.trapz(P_full, dust, axis=4)
    
    Pz = np.trapz(np.trapz(np.trapz(P.T, metal, axis=3), age, axis=2), tau, axis=1) 

    P = np.trapz(P, rshift, axis=3)
   
    PZ = np.trapz(np.trapz(P, tau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, tau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return PZ, Pt, Ptau, Pz, Pd

def Full_analyze(chi, mchi, metal, age, tau, rshift, dust, v, mod):
    lwa = np.load(temp_out + 'tmp_lwa_scale_{0}_v{1}.npy'.format(v, mod))
    
    ####### Get maximum age
    max_age = Oldest_galaxy(max(rshift))
    
    chi[ : , len(age[age <= max_age]):] = 1E15
    
    ######## get Pd and Pz 
    P_full = np.exp(- (chi - mchi) / 2).astype(np.float128)
    
    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, metal, axis=4), age, axis=3), tau, axis=2), rshift, axis=1) 

    P = np.trapz(P_full, dust, axis=4)
    
    Pz = np.trapz(np.trapz(np.trapz(P.T, metal, axis=3), age, axis=2), tau, axis=1) 

    P = np.trapz(P, rshift, axis=3)
       
    ####### Get scaling factor for tau reshaping   
    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > lwa.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    new_P = np.zeros(P.T.shape)

    ######## Reshape likelihood to get light weighted age instead of age when marginalized
    for i in range(len(tau)):
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(lwa.T[i].T[ii],P.T[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(0, overhead[i][ii]))
        new_P[i] = frame.T

    P = new_P.T

    # test_prob = np.trapz(test_P, ultau, axis=2)
    C = np.trapz(np.trapz(np.trapz(P, tau, axis=2), age, axis=1), metal)

    P /= C
    
    PZ = np.trapz(np.trapz(P, tau, axis=2), age, axis=1)
    Pt = np.trapz(np.trapz(P, tau, axis=2).T, metal, axis=1)
    Ptau = np.trapz(np.trapz(P.T, metal, axis=2), age, axis=1)

    return PZ, Pt, Ptau, Pz, Pd

def Redshift_analyze(chi, mchi, metal, age, tau, rshift, dust):
    ######## get Pd and Pz 
    P_full = np.exp(- (chi - mchi) / 2).astype(np.float128)
    
    return np.trapz(np.trapz(np.trapz(np.trapz(P_full, dust, axis=4).T, metal, axis=3), age, axis=2), tau, axis=1) 