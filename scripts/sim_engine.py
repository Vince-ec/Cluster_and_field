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

hpath = os.environ['HOME'] + '/'

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
    
    
def load_spec(field, galaxy_id, instr, lims, specz, grism = True):
    # if loading photometry FLT stands in for num
        
    if grism:
        W, F, E, FLT = np.load(spec_path + '{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))

        M = np.load(spec_path + 'spec_mask/{0}_{1}_{2}_mask.npy'.format(field, galaxy_id, instr))
        
        W = W[M]
        FLT = FLT[M]
        F = F[M]
        E = E[M] 
    
        IDX = [U for U in range(len(W)) if lims[0] <= W[U] <= lims[-1] and F[U]**2 > 0]

        W = W[IDX]
        WRF = W / (1 + specz)
        FLT = FLT[IDX]
        F = F[IDX] 
        E = E[IDX] 
        
        return W, WRF, F, E, FLT, IDX

    else:
        W, F, E, FLT = np.load(phot_path + '{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))
        
        WRF = W / (1 + specz)
        
        return W, WRF, F, E, FLT

def load_phot_precalc(Pnum):
    MDF = pd.read_pickle(phot_path + 'model_photometry_list.pkl')

    IDP = []
    for i in range(len(Pnum)):
        for ii in range(len(MDF)):
            if Pnum[i] == MDF.tmp_num[MDF.index[ii]]:
                IDP.append(ii)

    ### Define precalculated terms for photometry
    SWV, TR = np.load(template_path + 'master_tmp.npy')
    B = np.load(template_path + 'bottom_precalc.npy')
    DNU = np.load(template_path + 'dnu_precalc.npy')
    ADJ = np.load(template_path + 'adj_precalc.npy')
    MFWV = np.load(template_path + 'effwv_precalc.npy') 
        
    return MDF, IDP, SWV, TR, B, DNU, ADJ, MFWV
        
def load_beams_and_trns(wv, beam):
    ### set beams
    Beam = model.BeamCutout(fits_file = beam)

    ### Set transmission curve
    model_wave, model_flux = np.load(model_path + 'm0.019_a2.0_dt0_spec.npy')
    W, F = forward_model_grism(Beam, model_wave, np.ones(len(model_wave)))
    trans = interp1d(W,F)(wv)       

    return Beam, trans

def apply_tmp_err(wv_rf, er,flx, tmp_err = True, pht_err = 0):
    WV,TEF = np.load(template_path + 'template_error_updt.npy')
    iTEF = interp1d(WV,TEF)(wv_rf)
    
    if not tmp_err:
        iTEF = 0
        
    return np.sqrt(er**2 + (iTEF*flx)**2 + (pht_err*flx)**2)

def init_sim(model_wave, model_fl, specz, mass, stellar_mass, bwv, rwv, bflx, rflx, pflx, berr, rerr, perr, phot_err,
            btrans, rtrans, bbeam, rbeam, IDP, sens_wv, b, dnu, adj, offset_limit):
    fl_sol = F_lam_per_M(model_fl * (1 + specz), model_wave, specz, 0, stellar_mass)

    f_lam = fl_sol * 10**(mass)

    Bsnr = np.abs(bflx / berr)
    Rsnr = np.abs(rflx / rerr)
    Psnr = np.abs(pflx / np.sqrt(perr**2 + (phot_err * pflx)**2))

    SPfl = forward_model_phot(model_wave*(1 + specz), f_lam, IDP, sens_wv, b, dnu, adj)
    SPer = SPfl / Psnr

    Bmw, Bmf = forward_model_grism(bbeam, model_wave, f_lam)
    Rmw, Rmf = forward_model_grism(rbeam, model_wave, f_lam)
    iBmf = interp1d(Bmw,Bmf)(bwv)       
    iRmf = interp1d(Rmw,Rmf)(rwv)     

    SBfl = iBmf / btrans
    SRfl = iRmf / rtrans

    SBer = SBfl / Bsnr
    SRer = SRfl / Rsnr

    SPflx = SPfl + np.random.normal(0, np.abs(SPer))
    SBflx = SBfl + np.random.normal(0, np.abs(SBer))
    SRflx = SRfl + np.random.normal(0, np.abs(SRer))

    offset_range = np.arange(- offset_limit,offset_limit + 0.001,0.001)
    offset_B = np.random.choice(offset_range,1) + 1
    offset_R = np.random.choice(offset_range,1) + 1  

    SBflx *= offset_B ; SBer *= offset_B
    SRflx *= offset_R ; SRer *= offset_R
    
    return SBflx, SBer, SRflx, SRer, SPflx, SPer

def forward_model_grism(BEAM, model_wave, model_flux):
    ### creates a model using an individual beam
    BEAM.beam.compute_model(spectrum_1d=[model_wave, model_flux])
    w, f, e = BEAM.beam.optimal_extract(BEAM.beam.model, bin=0)
    return w, f

def forward_model_phot(model_wave, model_flux, IDP, sens_wv, b, dnu, adj):
    c = 3E18

    imfl =interp1d(c / model_wave, (c/(c / model_wave)**2) * model_flux)

    mphot = (np.trapz(imfl(c /(sens_wv[IDP])).reshape([len(IDP),len(sens_wv[0])]) \
                      * b[IDP], dnu[IDP])/np.trapz(b[IDP], dnu[IDP])) * adj[IDP]
    return np.array(mphot)

def Calzetti_low(Av,lam):
    lam = lam * 1E-4
    Rv=4.05
    k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
    cal = 10**(-0.4*k*Av/Rv)
    return cal

def Calzetti_hi(Av,lam):
    lam = lam * 1E-4
    Rv=4.05
    k = 2.659*(-1.857 +1.04/(lam)) + Rv
    cal = 10**(-0.4*k*Av/Rv)    
    
    return cal

def Calzetti(Av,lam):
    dust = Calzetti_low(Av,lam)
    dust2 = Calzetti_hi(Av,lam)
    
    for ii in range(len(dust)):
        if lam[ii] > 6300:
            dust[ii]=dust2[ii] 
    
    return dust
  
def Salmon_low(Av,lam):
    lam = lam * 1E-4
    lamv = 5510 * 1E-4
    Rv=4.05
    delta = 0.62 * np.log10(Av/Rv) + 0.26
    k = 2.659*(-2.156 +1.509/(lam) -0.198/(lam**2) +0.011/(lam**3)) + Rv
    sal = 10**(-0.4*k*(lam / lamv)**(delta)*Av/Rv)
    return sal

def Salmon_hi(Av,lam):
    lam = lam * 1E-4
    lamv = 5510 * 1E-4
    Rv=4.05
    delta = 0.62 * np.log10(Av/Rv) + 0.26
    k = 2.659*(-1.857 +1.04/(lam)) + Rv
    sal = 10**(-0.4*k*(lam / lamv)**(delta)*Av/Rv)    
    return sal

def Salmon(Av,lam):
    dust = Salmon_low(Av,lam)
    dust2 = Salmon_hi(Av,lam)
    
    for ii in range(len(dust)):
        if lam[ii] > 6300:
            dust[ii]=dust2[ii] 
    if Av == 0:
        dust = np.ones(len(dust))
    return dust
    
def Chi_Squared(data, model, error):
    return np.sum(((data-model) / error)**2)

def L_nu_per_M(l_aa, lam, z, Av, m_star):
    c = 3E18 # speed of light in angstrom
    lam_0 = lam / (1 + z) # restframe wavelenth in angstrom
    dust = 10**(-0.4*Av)
    return ((lam_0**2)/(c * m_star)) * l_aa * dust * 3.839E33

def F_nu_per_M(l_aa, lam, z, Av, m_star):
    conv = 3.086E24 # conversion of Mpc to cm
    D_l = cosmo.luminosity_distance(z).value # in Mpc
    return (1 + z) * L_nu_per_M(l_aa, lam, z, Av, m_star)  / (4 * np.pi * (D_l*conv)**2)

def F_lam_per_M(l_aa, lam, z, Av, m_star):
    c = 3E18 # speed of light in angstrom
    return (c / lam**2) * F_nu_per_M(l_aa, lam, z, Av, m_star)

def Get_mass(gwv, gfl, ger, Z, t, z, Av):
    sp = fsps.StellarPopulation(imf_type=1, tpagb_norm_type=0, zcontinuous=1, logzsol=np.log10(Z / 0.019), sfh=0)
    wave,flux=np.array(sp.get_spectrum(tage=t,peraa=True))
    
    fl_m = F_lam_per_M(flux, wave * (1 + z), z, Av, sp.stellar_mass)
    
    IDX = [U for U in range(len(gwv)) if 8000 < gwv[U] < 11300]
    return np.log10(Scale_model(gfl[IDX],ger[IDX],interp1d(wv,fl_m)(gwv[IDX])))