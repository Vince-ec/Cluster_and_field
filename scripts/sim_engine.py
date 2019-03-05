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
    
"""
def:
-load_spec
-load_phot_precalc
-load_beams_and_trns
-apply_tmp_err
-init_sim
-forward_model_grism
-forward_model_phot
-Calzetti_low
-Calzetti_hi
-Calzetti
-Salmon_low
-Salmon_hi
-Salmon
-Chi_Squared
-L_nu_per_M
-F_nu_per_M
-F_lam_per_M
-Get_mass
"""  

def load_spec(field, galaxy_id, instr, lims, specz, grism = True, trim = None):
    # if loading photometry FLT stands in for num
    bfilters = [34, 36, 37, 58, 117, 118, 195, 196, 220, 224]

    if grism:
        W, F, E, FLT, L, C = np.load(spec_path + '{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))
    
        IDX = [U for U in range(len(W)) if lims[0] <= W[U] <= lims[-1] and F[U]**2 > 0]

        W = np.array(W[IDX])
        WRF = np.array(W / (1 + specz))
        FLT = np.array(FLT[IDX])
        F = np.array(F[IDX]) 
        E = np.array(E[IDX]) 
        L = np.array(L[IDX]) 
        C = np.array(C[IDX]) 
        
        if trim == None:
            trim = 0 

        return W[WRF > trim], WRF[WRF > trim], F[WRF > trim], E[WRF > trim], FLT[WRF > trim], np.array(IDX)[WRF > trim], L[WRF > trim], C[WRF > trim]

    else:
        W, F, E, FLT = np.load(phot_path + '{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))
        
        WRF = W / (1 + specz)
        
        IDX = []
        
        for i in range(len(FLT)):
            if FLT[i] not in bfilters and F[i] / E[i] > 0.5:
                IDX.append(i)
        
        W, WRF, F, E, FLT = W[IDX], WRF[IDX], F[IDX], E[IDX], FLT[IDX]
        
        W, WRF, F, E, FLT = W[F > 0], WRF[F > 0], F[F > 0], E[F > 0], FLT[F > 0]
        
        if trim == None:
            trim = 0 
        
        return W[WRF > trim], WRF[WRF > trim], F[WRF > trim], E[WRF > trim], FLT[WRF > trim]
        
        
        
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
    ### Set transmission curve
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.002/0.019), 
                                sfh = 4, tau = 0.6, dust_type = 1)

    model_wave, model_flux = sp.get_spectrum(tage = 3.6, peraa = True)

    ### set beams
    BEAMS = []
    TRANS = []
    
    for i in beam:
        Beam = model.BeamCutout(fits_file = i)

        W, F = forward_model_grism(Beam, model_wave, np.ones(len(model_wave)))
        trans = interp1d(W,F)(wv)       
        
        BEAMS.append(Beam)
        TRANS.append(trans)
        
    return BEAMS, TRANS

def apply_tmp_err(wv, wv_rf, er, flx, instr, mdl_err = True):
    
    if mdl_err:
        if instr == 'P':
            WV_RF, MEF = np.load(template_path + 'P_mdl_EF.npy')
            WV, IEF = np.load(template_path + 'P_inst_EF.npy')
            
            iMEF = interp1d(WV_RF,MEF)(wv_rf)
            iIEF = interp1d(WV,IEF)(wv)

        if instr == 'R':
            WV_RF, MEF = np.load(template_path + 'R_mdl_EF.npy')
            WV, IEF = np.load(template_path + 'R_inst_EF.npy')
            
            iMEF = interp1d(WV_RF,MEF)(wv_rf)
            iIEF = interp1d(WV,IEF)(wv)
            
        if instr == 'B':
            iMEF =0
            iIEF = 0
    
        er = np.sqrt(er**2 + (0.5 * iMEF*flx)**2 + (0.5 * iIEF*flx)**2)

        
    return er

#def init_sim(model_wave, model_fl, specz, bwv, rwv, bflx, rflx, pflx, berr, rerr, perr, phot_err,
#            btrans, rtrans, bflat, rflat, bbeam, rbeam, IDP, sens_wv, b, dnu, adj):
def init_sim(model_wave, model_fl, specz, bwv, rwv, bflx, rflx, pflx, berr, rerr, perr, phot_err,
            btrans, rtrans, bbeam, rbeam, IDP, sens_wv, b, dnu, adj): 
    # convert model to f_lam / sol_mass
    fl_sol = F_lam_per_M(model_fl * (1 + specz), model_wave, specz, 0, 1)

    # make models
    SPfl = forward_model_phot(model_wave*(1 + specz), fl_sol, IDP, sens_wv, b, dnu, adj)

    Bmf = forward_model_all_beams_flatted(bbeam, btrans, bwv, model_wave*(1 + specz), fl_sol)
    Rmf = forward_model_all_beams_flatted(rbeam, rtrans, rwv, model_wave*(1 + specz), fl_sol)
    
    SPerr = perr
    SBer = berr
    SRer = rerr
    
    mass = Scale_model(pflx, perr, SPfl)

    
    SPflx = mass * SPfl
    SBfl = mass * Bmf
    SRfl = mass * Rmf
    
    # scale by mass
    #mass = Scale_model(pflx, perr, SPfl)
    #logmass = np.log10(mass)
    
    #SPflx = SPfl * mass
    
    #SBer = SBerr / bflat
    #SRer = SRerr / rflat

    SPflx = SPflx + np.random.normal(0, np.abs(SPerr))
    SBfl = SBfl + np.random.normal(0, np.abs(SBer))
    SRfl = SRfl + np.random.normal(0, np.abs(SRer))
    
    #SBfl = (mass / Scale_model(bflx,berr,Bmf)) * SBflx / btrans
    #SRfl = (mass / Scale_model(rflx,rerr,Rmf)) * SRflx / rtrans
  
    #return SBflx, SBerr, SBfl, SBer, SRflx, SRerr, SRfl, SRer, SPflx, SPerr, mass, logmass
    return SBfl, SBer, SRfl, SRer, SPflx, SPerr

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

def forward_model_all_beams(beams, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)

    return np.mean(FL.T,axis=1)

def forward_model_all_beams_flatted(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def decontaminate(W, WRF, F, E, FLT, IDX, L, C):
    IDC = []
        
    for i in range(len(W)):
        if (C[i] + E[i]) > L[i]:
            IDC.append(i)

    W = W[IDC]
    WRF = WRF[IDC]
    F = F[IDC]
    E = E[IDC]
    FLT = FLT[IDC]
    IDX = np.array(IDX)[IDC]
    L = L[IDC]
    C = C[IDC]
    
    return W, WRF, F, E, FLT, IDX, L, C


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
    sp = fsps.StellarPopulation(imf_type = 0, tpagb_norm_type = 0, zcontinuous = 1, logzsol = np.log10(Z/0.019), 
                                sfh = 4, tau = 0.6, dust_type = 1)
    
    wave,flux=np.array(sp.get_spectrum(tage=t,peraa=True))
    
    fl_m = F_lam_per_M(flux, wave * (1 + z), z, Av, sp.stellar_mass)
    
    IDX = [U for U in range(len(gwv)) if 8000 < gwv[U] < 11300]
    return np.log10(Scale_model(gfl[IDX],ger[IDX],interp1d(wv,fl_m)(gwv[IDX])))

