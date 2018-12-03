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

"""
class:
Gen_spec

def:
Calzetti_low
Calzetti_hi
Calzetti
Chisquared
Gen_dust_minigrid
Stitch_spec
Resize
Redden_and_fit   
Stitch_resize_redden_fit
Stich_grids
Analyze_full_fit
Fit_all
Fit_rshift
Analyze_indv_chi
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
  
def Chi_Squared(data, model, error):
    return np.sum(((data-model) / error)**2)
    

class Gen_spec(object):
    def __init__(self, field, galaxy_id, specz, g102_beam, g141_beam,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                 filter_102 = 201, filter_141 = 203, tmp_err = False, 
                 phot_tmp_err = False, errterm = 0):
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
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        tmp_err - (flag) whether or not we apply a template error function (not available)
        """
        self.Bwv, self.Bflx, self.Berr, self.Bflt = np.load(spec_path + '{0}_{1}_g102.npy'.format(field, galaxy_id))
        self.Rwv, self.Rflx, self.Rerr, self.Rflt = np.load(spec_path + '{0}_{1}_g141.npy'.format(field, galaxy_id))
        self.Bmask = np.load(spec_path + 'spec_mask/{0}_{1}_g102_mask.npy'.format(field, galaxy_id))
        self.Rmask = np.load(spec_path + 'spec_mask/{0}_{1}_g141_mask.npy'.format(field, galaxy_id))
        
        self.Bwv = self.Bwv[self.Bmask]
        self.Bflt = self.Bflt[self.Bmask]
        self.Bflx = self.Bflx[self.Bmask]
        self.Berr = self.Berr[self.Bmask] 
        
        self.Rwv = self.Rwv[self.Rmask]
        self.Rflt = self.Rflt[self.Rmask]
        self.Rflx = self.Rflx[self.Rmask]
        self.Rerr = self.Rerr[self.Rmask] 
        
        self.Pwv, self.Pflx, self.Perr, self.Pnum = np.load(phot_path + '{0}_{1}_phot.npy'.format(field, galaxy_id))
        self.Pwv_rf = self.Pwv / (1 + self.specz)
                
        self.IDB = [U for U in range(len(self.Bwv)) if g102_lims[0] <= self.Bwv[U] <= g102_lims[-1] and self.Bflx[U]**2 > 0]
        self.IDR = [U for U in range(len(self.Rwv)) if g141_lims[0] <= self.Rwv[U] <= g141_lims[-1] and self.Rflx[U]**2 > 0]

        self.Bwv = self.Bwv[self.IDB]
        self.Bwv_rf = self.Bwv / (1 + specz)
        self.Bflt = self.Bflt[self.IDB]
        self.Bflx = self.Bflx[self.IDB] #* Bscale
        self.Berr = self.Berr[self.IDB] #* Bscale
        
        self.Rwv = self.Rwv[self.IDR]
        self.Rwv_rf = self.Rwv / (1 + specz)
        self.Rflt = self.Rflt[self.IDR]
        self.Rflx = self.Rflx[self.IDR] #* Rscale
        self.Rerr = self.Rerr[self.IDR] #* Rscale

        self.model_photDF = pd.read_pickle(phot_path + 'model_photometry_list.pkl')
        
        self.IDP = []
        for i in range(len(self.Pnum)):
            for ii in range(len(self.model_photDF)):
                if self.Pnum[i] == self.model_photDF.tmp_num[self.model_photDF.index[ii]]:
                    self.IDP.append(ii)
        
        if phot_tmp_err:
            ewv, tmp= np.loadtxt(template_path + 'TEMPLATE_ERROR.eazy_v1.0').T
            iphterr = interp1d(ewv,tmp)(self.Pwv_rf)
            self.Perr_o = self.Perr
            self.Perr = np.sqrt(self.Perr**2 + (iphterr * self.Pflx)**2+ (errterm * self.Pflx)**2)
            
#         if tmp_err:
#             WV,TEF = np.load(data_path + 'template_error_function.npy')
#             iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
#             self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.fl)**2)

        self.Bbeam = model.BeamCutout(fits_file = g102_beam)
        self.Rbeam = model.BeamCutout(fits_file = g141_beam)

        #self.Bpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g102.A.fits')
        #self.Rpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g141.A.fits')
        
        ### Define precalculated terms for photometry
        self.sens_wv, self.trans = np.load(template_path + 'master_tmp.npy')
        self.b = np.load(template_path + 'bottom_precalc.npy')
        self.dnu = np.load(template_path + 'dnu_precalc.npy')
        self.adj = np.load(template_path + 'adj_precalc.npy')
        self.mdleffwv = np.load(template_path + 'effwv_precalc.npy') 
    
    def Sim_spec_indv(self, BEAM, model_wave, model_flux):
        ### creates a model using an individual beam
        BEAM.beam.compute_model(spectrum_1d=[model_wave, model_flux])
        w, f, e = BEAM.beam.optimal_extract(BEAM.beam.model, bin=0)
        return w, f
        
    def Sim_spec_mult(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bbeam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rbeam, model_wave, model_flux)

    def Sim_spec_mult_point(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bpoint_beam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rpoint_beam, model_wave, model_flux)
    
    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))

        [Bmw, Bmf], [Rmw, Rmf] = self.Sim_spec_mult(model_wave * (1 + model_redshift), 
                                                                        model_flux * Calzetti(Av,model_wave))
        iBmf = interp1d(Bmw,Bmf)(self.Bwv)       
        iRmf = interp1d(Rmw,Rmf)(self.Rwv)     
        
        self.Bmfl = iBmf / self.Bflt
        self.Rmfl = iRmf / self.Rflt
            
        self.Bmfl *= Scale_model(self.Bflx, self.Berr, self.Bmfl)
        self.Rmfl *= Scale_model(self.Rflx, self.Rerr, self.Rmfl)
        
        if multi_component:
            [Bpmw, Bpmf], [Rpmw, Rpmf] = self.Sim_spec_mult_point(model_wave * (1 + model_redshift), 
                                                                            model_flux * Calzetti(Av,model_wave))
            iBpmf = interp1d(Bpmw,Bpmf)(self.Bwv)       
            iRpmf = interp1d(Rpmw,Rpmf)(self.Rwv)     

            self.Bpmfl = iBpmf / self.Bflt
            self.Rpmfl = iRpmf / self.Rflt

            self.Bpmfl *= Scale_model(self.Bflx, self.Berr, self.Bpmfl)
            self.Rpmfl *= Scale_model(self.Rflx, self.Rerr, self.Rpmfl)
            
            self.Bpmfl *= point_scale
            self.Rpmfl *= point_scale
            
            self.BMCmfl = self.Bmfl + self.Bpmfl
            self.RMCmfl = self.Rmfl + self.Rpmfl
            
            self.BMCmfl *= Scale_model(self.Bflx, self.Berr, self.BMCmfl)
            self.RMCmfl *= Scale_model(self.Rflx, self.Rerr, self.RMCmfl)
       
    def Sim_phot_mult(self, model_wave, model_flux):
        
        imfl =interp1d(self.c / model_wave, (self.c/(self.c / model_wave)**2) * model_flux)

        mphot = (np.trapz(imfl(self.c /(self.sens_wv[self.IDP])).reshape([len(self.IDP),len(self.sens_wv[0])]) \
                          * self.b[self.IDP],self.dnu[self.IDP])/np.trapz(self.b[self.IDP],
                                                                          self.dnu[self.IDP])) * self.adj[self.IDP]
        
        return np.array([self.mdleffwv[self.IDP],mphot])

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))
        
        self.Pmwv, self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), 
                                                  model_flux * Calzetti(Av,model_wave))
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        self.Sim_phot(metal, age, tau, model_redshift, Av)
    
    
    
def Scale_model_mult(D, sig, M):
    C = np.sum(((D * M) / sig ** 2), axis=1) / np.sum((M ** 2 / sig ** 2), axis=1)
    return C

def Gen_mflgrid(spec, name, metal, age, tau, rshift):
    wv,fl = np.load(model_path + 'm0.019_a2.8_dt0_spec.npy')

    [Bmwv,Bmf_len], [Rmwv,Rmf_len] = spec.Sim_spec_mult(wv,fl)
    
    ##### set model wave
    for i in range(len(metal)):
        
        Bmfl = np.zeros([len(age)*len(tau)*len(rshift),len(Bmf_len)])
        Rmfl = np.zeros([len(age)*len(tau)*len(rshift),len(Rmf_len)])
        Pmfl = np.zeros([len(age)*len(tau)*len(rshift),len(spec.IDP)])

        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(metal[i], age[ii], tau[iii]))
                for iv in range(len(rshift)):
                    [Bmwv,Bmflx], [Rmwv,Rmflx] = spec.Sim_spec_mult(wv * (1 + rshift[iv]),fl)
                    Pmwv, Pmflx = spec.Sim_phot_mult(wv * (1 + rshift[iv]),fl)
                                      
                    Bmfl[ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Bmflx
                    Rmfl[ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Rmflx
                    Pmfl[ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Pmflx
        
        np.save(chi_path + 'spec_files/{0}_m{1}_g102'.format(name, metal[i]),Bmfl)
        np.save(chi_path + 'spec_files/{0}_m{1}_g141'.format(name, metal[i]),Rmfl)
        np.save(chi_path + 'spec_files/{0}_m{1}_phot'.format(name, metal[i]),Pmfl)

#####
#####     
    
def Gen_dust_minigrid(fit_wv,rshift):
    dust_dict = {}
    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        key = str(Av[i])
        minigrid = np.zeros([len(rshift),len(fit_wv)])
        for ii in range(len(rshift)):
            minigrid[ii] = Calzetti(Av[i],fit_wv / (1 + rshift[ii]))
        dust_dict[key] = minigrid
    return dust_dict
         
def Stitch_spec(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))
        
    stc = np.array(stc)
    return stc.reshape([stc.shape[0] * stc.shape[1],stc.shape[2]])

def Resize(fit_wv, fit_flat, mwv, mfl):
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(mwv,range(len(mfl.data)),mfl.data)(fit_wv,range(len(mfl.data)))
    return mfl / fit_flat
    
def Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname):    
    minidust = Gen_dust_minigrid(fit_wv,redshift)

    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
        redflgrid = mfl * dustgrid
        SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
        redflgrid = np.array([SCL]).T*redflgrid
        chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
        np.save(chi_path + '{0}_d{1}_{2}_chidata'.format(outname, i, instrument),chigrid)
        
def Stitch_resize_redden_fit(fit_wv, fit_fl, fit_er, fit_flat, instrument, name, mwv, 
                     metal, age, tau, redshift, outname,resize=True):
    #############Read in spectra and stich spectra grid together#################
    files = [chi_path + 'spec_files/{0}_m{1}_{2}.npy'.format(name, U, instrument) for U in metal]
    mfl = Stitch_spec(files)
    
    if resize:
        mfl = Resize(fit_wv, fit_flat, mwv, mfl)

    Redden_and_fit(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname)  

def Stich_grids(grids):
    stc = []
    for i in range(len(grids)):
        stc.append(np.load(grids[i]))
    return np.array(stc)
    
def Analyze_full_fit(outname, metal, age, tau, rshift, dust = np.arange(0,1.1,0.1), age_conv=data_path + 'light_weight_scaling_3.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(max(rshift))
    
    ####### Read in file   
    chi = np.zeros([len(dust),len(metal),len(age),len(tau),len(rshift)])
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
    P_full = np.exp(- chi / 2).astype(np.float128)

    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1),dust)

    Pz = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1),rshift)

    P = np.trapz(P_full, rshift, axis=4)
    P = np.trapz(P.T, dust, axis=3).T
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

    return prob.T, PZ, Pt, Ptau, Pz, Pd

def Fit_all(field, galaxy, g102_beam, g141_beam, specz, metal, age, tau, rshift, dust, name, 
        gen_models = True, age_conv= data_path + 'light_weight_scaling_3.npy', errterm = 0,
           outname = 'none'):
   
    if outname == 'none':
        outname = name
    ######## initialize spec
    sp = Gen_spec(field, galaxy, specz, g102_beam, g141_beam, phot_tmp_err = True, errterm = errterm)
    
    if gen_models:
        Gen_mflgrid(sp, name, metal, age, tau, rshift)

    ## set some variables
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv,fl)
    
    Stitch_resize_redden_fit(sp.Bwv, sp.Bflx, sp.Berr, sp.Bflt, 'g102', name, Bmwv, 
                     metal, age, tau, rshift, outname)
    Stitch_resize_redden_fit(sp.Rwv, sp.Rflx, sp.Rerr, sp.Rflt, 'g141', name, Rmwv, 
                     metal, age, tau, rshift, outname)
    Stitch_resize_redden_fit(sp.Pwv, sp.Pflx, sp.Perr, 'none', 'phot', name, sp.Pwv, 
                     metal, age, tau, rshift, outname, resize = False) 
    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(outname, metal, age, tau, rshift, 
                                               dust=dust,age_conv = age_conv)

    np.save(out_path + '{0}_tZ_pos'.format(outname),P)
    np.save(out_path + '{0}_Z_pos'.format(outname),[metal,PZ])
    np.save(out_path + '{0}_t_pos'.format(outname),[age,Pt])
    np.save(out_path + '{0}_tau_pos'.format(outname),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_rs_pos'.format(outname),[rshift,Pz])
    np.save(out_path + '{0}_d_pos'.format(outname),[dust,Pd])

def Analyze_indv_chi(outname, metal, age, tau, rshift, dust = np.arange(0,1.1,0.1),
                     age_conv=data_path + 'light_weight_scaling_3.npy', instr = 'none'):
    ####### Get maximum age
    max_age = Oldest_galaxy(max(rshift))
    
    ####### Read in file       
    chifiles = [chi_path + '{0}_d{1}_{2}_chidata.npy'.format(outname, U, instr) for U in range(len(dust))]
    chi = Stich_grids(chifiles)
    
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
    P_full = np.exp(- chi / 2).astype(np.float128)

    Pd = np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full, rshift, axis=4), ultau, axis=3), age, axis=2), metal, axis=1),dust)

    Pz = np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1) /\
        np.trapz(np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, dust, axis=4), metal, axis=3), age, axis=2), ultau, axis=1),rshift)

    P = np.trapz(P_full, rshift, axis=4)
    P = np.trapz(P.T, dust, axis=3).T
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

    np.save(out_path + '{0}_tZ_{1}_pos'.format(outname, instr),prob.T)
    np.save(out_path + '{0}_Z_{1}_pos'.format(outname, instr),[metal,PZ])
    np.save(out_path + '{0}_t_{1}_pos'.format(outname, instr),[age,Pt])
    np.save(out_path + '{0}_tau_{1}_pos'.format(outname, instr),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_rs_{1}_pos'.format(outname, instr),[rshift,Pz])
    np.save(out_path + '{0}_d_{1}_pos'.format(outname, instr),[dust,Pd])
      
def Resize_and_fit(fit_wv, fit_fl, fit_er, fit_flat, mwv, mfl, metal, age, tau, rshift):
    mfl = np.ma.masked_invalid(mfl)
    mfl.data[mfl.mask] = 0
    mfl = interp2d(mwv,range(len(mfl.data)),mfl.data)(fit_wv,range(len(mfl.data)))
    mfl = mfl / fit_flat
    SCL = Scale_model_mult(fit_fl,fit_er,mfl)
    return np.sum(((fit_fl - np.array([SCL]).T*mfl) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)])

def Analyze_rshift(chi, metal, age, tau, rshift):
    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    ######## get Pd and Pz 
    P_full = np.exp(- chi / 2).astype(np.float128)

    Pz = np.trapz(np.trapz(np.trapz(P_full.T, metal, axis=3), age, axis=2), ultau, axis=1) / \
         np.trapz(np.trapz(np.trapz(np.trapz(P_full.T, metal, axis=3), age, axis=2), ultau, axis=1),rshift)

    return Pz

def Fit_rshift(field, galaxy, g102_beam, g141_beam, metal, age, tau, rshift, errterm = 0):
    ######## initialize spec
    sp = Gen_spec(field, galaxy, 1, g102_beam, g141_beam, g102_lims=[8500,11300], phot_tmp_err = True, errterm = errterm)
    
    wv,fl = np.load(model_path + 'm0.019_a2.8_dt0_spec.npy')
    [Bmwv,Bmf_len], [Rmwv,Rmf_len] = sp.Sim_spec_mult(wv,fl)
    
    ##### set model wave
    Bmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(Bmf_len)])
    Rmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(Rmf_len)])
    Pmfl = np.zeros([len(metal)*len(age)*len(tau)*len(rshift),len(sp.IDP)])
    
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                wv,fl = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(metal[i], age[ii], tau[iii]))
                for iv in range(len(rshift)):
                    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv * (1 + rshift[iv]),fl)
                    Pmwv, Pmflx = sp.Sim_phot_mult(wv * (1 + rshift[iv]),fl)
                                      
                    Bmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Bmflx
                    Rmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Rmflx
                    Pmfl[i*len(age)*len(tau)*len(rshift) + ii*len(tau)*len(rshift) + iii*len(rshift) + iv] = Pmflx

    ## set some variables
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv,fl)
    
    Bchi = Resize_and_fit(sp.Bwv, sp.Bflx, sp.Berr, sp.Bflt, Bmwv, Bmfl, metal, age, tau, rshift)
    Rchi = Resize_and_fit(sp.Rwv, sp.Rflx, sp.Rerr, sp.Rflt, Rmwv, Rmfl, metal, age, tau, rshift)
        
    SCL = Scale_model_mult(sp.Pflx, sp.Perr, Pmfl)
    Pchi = np.sum(((sp.Pflx - np.array([SCL]).T*Pmfl) / sp.Perr) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(rshift)])
    
    Achi = Bchi + Rchi + Pchi
    
    Pz = Analyze_rshift(Achi, metal, age, tau, rshift)
    Pzb = Analyze_rshift(Bchi, metal, age, tau, rshift)
    Pzr = Analyze_rshift(Rchi, metal, age, tau, rshift)
    Pzp = Analyze_rshift(Pchi, metal, age, tau, rshift)
    
    np.save(out_path + '{0}_{1}_rs_lres_pos'.format(field, galaxy),[rshift, Pz, Pzb, Pzr, Pzp])

###################################################
###################################################

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

class Gen_spec2(object):
    def __init__(self, field, galaxy_id, specz, g102_beam, g141_beam,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                 filter_102 = 201, filter_141 = 203, tmp_err = False, 
                 phot_tmp_err = False, errterm = 0):
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
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        tmp_err - (flag) whether or not we apply a template error function (not available)
        """
        self.Bwv, self.Bflx, self.Berr, self.Bflt = np.load(spec_path + '{0}_{1}_g102.npy'.format(field, galaxy_id))
        self.Rwv, self.Rflx, self.Rerr, self.Rflt = np.load(spec_path + '{0}_{1}_g141.npy'.format(field, galaxy_id))
        self.Bmask = np.load(spec_path + 'spec_mask/{0}_{1}_g102_mask.npy'.format(field, galaxy_id))
        self.Rmask = np.load(spec_path + 'spec_mask/{0}_{1}_g141_mask.npy'.format(field, galaxy_id))
        
        self.Bwv = self.Bwv[self.Bmask]
        self.Bflt = self.Bflt[self.Bmask]
        self.Bflx = self.Bflx[self.Bmask]
        self.Berr = self.Berr[self.Bmask] 
        
        self.Rwv = self.Rwv[self.Rmask]
        self.Rflt = self.Rflt[self.Rmask]
        self.Rflx = self.Rflx[self.Rmask]
        self.Rerr = self.Rerr[self.Rmask] 
        
        self.Pwv, self.Pflx, self.Perr, self.Pnum = np.load(phot_path + '{0}_{1}_phot.npy'.format(field, galaxy_id))
        self.Pwv_rf = self.Pwv / (1 + self.specz)
                
        self.IDB = [U for U in range(len(self.Bwv)) if g102_lims[0] <= self.Bwv[U] <= g102_lims[-1] and self.Bflx[U]**2 > 0]
        self.IDR = [U for U in range(len(self.Rwv)) if g141_lims[0] <= self.Rwv[U] <= g141_lims[-1] and self.Rflx[U]**2 > 0]

        self.Bwv = self.Bwv[self.IDB]
        self.Bwv_rf = self.Bwv / (1 + specz)
        self.Bflt = self.Bflt[self.IDB]
        self.Bflx = self.Bflx[self.IDB] #* Bscale
        self.Berr = self.Berr[self.IDB] #* Bscale
        
        self.Rwv = self.Rwv[self.IDR]
        self.Rwv_rf = self.Rwv / (1 + specz)
        self.Rflt = self.Rflt[self.IDR]
        self.Rflx = self.Rflx[self.IDR] #* Rscale
        self.Rerr = self.Rerr[self.IDR] #* Rscale

        self.model_photDF = pd.read_pickle(phot_path + 'model_photometry_list.pkl')
        
        self.IDP = []
        for i in range(len(self.Pnum)):
            for ii in range(len(self.model_photDF)):
                if self.Pnum[i] == self.model_photDF.tmp_num[self.model_photDF.index[ii]]:
                    self.IDP.append(ii)
        
        if phot_tmp_err:
            ewv, tmp= np.loadtxt(template_path + 'TEMPLATE_ERROR.eazy_v1.0').T
            iphterr = interp1d(ewv,tmp)(self.Pwv_rf)
            self.Perr_o = self.Perr
            self.Perr = np.sqrt(self.Perr**2 + (iphterr * self.Pflx)**2+ (errterm * self.Pflx)**2)
            
#         if tmp_err:
#             WV,TEF = np.load(data_path + 'template_error_function.npy')
#             iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
#             self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.fl)**2)

        self.Bbeam = model.BeamCutout(fits_file = g102_beam)
        self.Rbeam = model.BeamCutout(fits_file = g141_beam)

        #self.Bpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g102.A.fits')
        #self.Rpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g141.A.fits')
        
        ### Define precalculated terms for photometry
        self.sens_wv, self.trans = np.load(template_path + 'master_tmp.npy')
        self.b = np.load(template_path + 'bottom_precalc.npy')
        self.dnu = np.load(template_path + 'dnu_precalc.npy')
        self.adj = np.load(template_path + 'adj_precalc.npy')
        self.mdleffwv = np.load(template_path + 'effwv_precalc.npy') 
    
        ### Set transmission curve
        model_wave, model_flux = np.load(model_path + 'm0.019_a2.0_dt0_spec.npy')
        Bw,Bfl = self.Sim_spec_indv(self.Bbeam, model_wave, np.ones(len(model_wave)))
        Rw,Rfl = self.Sim_spec_indv(self.Rbeam, model_wave, np.ones(len(model_wave)))
        self.Btrans = interp1d(Bw,Bfl)(self.Bwv)       
        self.Rtrans = interp1d(Rw,Rfl)(self.Rwv) 
    
    def Sim_spec_indv(self, BEAM, model_wave, model_flux):
        ### creates a model using an individual beam
        BEAM.beam.compute_model(spectrum_1d=[model_wave, model_flux])
        w, f, e = BEAM.beam.optimal_extract(BEAM.beam.model, bin=0)
        return w, f
        
    def Sim_spec_mult(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bbeam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rbeam, model_wave, model_flux)

    def Sim_spec_mult_point(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bpoint_beam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rpoint_beam, model_wave, model_flux)
    
    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))

        [Bmw, Bmf], [Rmw, Rmf] = self.Sim_spec_mult(model_wave * (1 + model_redshift), 
                                                                        model_flux * Calzetti(Av,model_wave))
        iBmf = interp1d(Bmw,Bmf)(self.Bwv)       
        iRmf = interp1d(Rmw,Rmf)(self.Rwv)     
        
        self.Bmfl = iBmf / self.Btrans
        self.Rmfl = iRmf / self.Rtrans
            
        self.Bmfl *= self.PC
        self.Rmfl *= self.PC
        
        Bscale = Scale_model(self.Bflx, self.Berr, self.Bmfl)
        Rscale = Scale_model(self.Rflx, self.Rerr, self.Rmfl)
        
        self.Bflx = self.Bflx / Bscale ; self.Berr = self.Berr / Bscale 
        self.Rflx = self.Rflx / Rscale ; self.Rerr = self.Rerr / Rscale 
        
        if multi_component:
            [Bpmw, Bpmf], [Rpmw, Rpmf] = self.Sim_spec_mult_point(model_wave * (1 + model_redshift), 
                                                                            model_flux * Calzetti(Av,model_wave))
            iBpmf = interp1d(Bpmw,Bpmf)(self.Bwv)       
            iRpmf = interp1d(Rpmw,Rpmf)(self.Rwv)     

            self.Bpmfl = iBpmf / self.Bflt
            self.Rpmfl = iRpmf / self.Rflt

            self.Bpmfl *= Scale_model(self.Bflx, self.Berr, self.Bpmfl)
            self.Rpmfl *= Scale_model(self.Rflx, self.Rerr, self.Rpmfl)
            
            self.Bpmfl *= point_scale
            self.Rpmfl *= point_scale
            
            self.BMCmfl = self.Bmfl + self.Bpmfl
            self.RMCmfl = self.Rmfl + self.Rpmfl
            
            self.BMCmfl *= Scale_model(self.Bflx, self.Berr, self.BMCmfl)
            self.RMCmfl *= Scale_model(self.Rflx, self.Rerr, self.RMCmfl)
       
    def Sim_phot_mult(self, model_wave, model_flux):
        
        imfl =interp1d(self.c / model_wave, (self.c/(self.c / model_wave)**2) * model_flux)

        mphot = (np.trapz(imfl(self.c /(self.sens_wv[self.IDP])).reshape([len(self.IDP),len(self.sens_wv[0])]) \
                          * self.b[self.IDP],self.dnu[self.IDP])/np.trapz(self.b[self.IDP],
                                                                          self.dnu[self.IDP])) * self.adj[self.IDP]
        
        return np.array([self.mdleffwv[self.IDP],mphot])

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))
        
        self.Pmwv, self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), 
                                                  model_flux * Calzetti(Av,model_wave))
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self, bfZ, bft, bftau, bfz, bfd):
        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(bfZ, bft, bftau))
        
        US_model_flux = F_lam_per_M(model_flux * Calzetti(bfd,model_wave), model_wave * (1 + bfz), bfz, 0, 1)

        US_pwv, US_pfl = self.Sim_phot_mult(model_wave * (1 + bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        Bw,Bf = self.Sim_spec_indv(self.Bbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        Rw,Rf = self.Sim_spec_indv(self.Rbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        
        iBmf = interp1d(Bw,Bf)(self.Bwv)       
        iRmf = interp1d(Rw,Rf)(self.Rwv)  
  
        Bmfl = iBmf / self.Btrans
        Rmfl = iRmf /self.Rtrans

        self.Bscale = Scale_model(self.Bflx, self.Berr, Bmfl)
        self.Rscale = Scale_model(self.Rflx, self.Rerr, Rmfl)
        
        self.Bflx = self.Bflx / self.Bscale ; self.Berr = self.Berr / self.Bscale 
        self.Rflx = self.Rflx / self.Rscale ; self.Rerr = self.Rerr / self.Rscale 

def Fit_all2(field, galaxy, g102_beam, g141_beam, specz, metal, age, tau, rshift, dust, name, 
        gen_models = True, age_conv= data_path + 'light_weight_scaling_3.npy', errterm = 0,
           outname = 'none'):
   
    if outname == 'none':
        outname = name
    ######## initialize spec
    sp = Gen_spec2(field, galaxy, specz, g102_beam, g141_beam, phot_tmp_err = True, errterm = errterm)
    
    if gen_models:
        Gen_mflgrid(sp, name, metal, age, tau, rshift)

    ## set some variables
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv,fl)
    
    Stitch_resize_redden_fit2(sp.Pwv, sp.Pflx, sp.Perr, 'none', 'phot', name, sp.Pwv, 
                     metal, age, tau, rshift, outname, phot = True) 
    
    Stitch_resize_redden_fit2(sp.Bwv, sp.Bflx, sp.Berr, sp.Btrans, 'g102', name, Bmwv, 
                     metal, age, tau, rshift, outname)
    Stitch_resize_redden_fit2(sp.Rwv, sp.Rflx, sp.Rerr, sp.Rtrans, 'g141', name, Rmwv, 
                     metal, age, tau, rshift, outname)

    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(outname, metal, age, tau, rshift, 
                                               dust=dust,age_conv = age_conv)

    np.save(out_path + '{0}_tZ_pos'.format(outname),P)
    np.save(out_path + '{0}_Z_pos'.format(outname),[metal,PZ])
    np.save(out_path + '{0}_t_pos'.format(outname),[age,Pt])
    np.save(out_path + '{0}_tau_pos'.format(outname),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_rs_pos'.format(outname),[rshift,Pz])
    np.save(out_path + '{0}_d_pos'.format(outname),[dust,Pd])


def Stitch_resize_redden_fit2(fit_wv, fit_fl, fit_er, fit_flat, instrument, name, mwv, 
                     metal, age, tau, redshift, outname, phot=False):
    #############Read in spectra and stich spectra grid together#################
    files = [chi_path + 'spec_files/{0}_m{1}_{2}.npy'.format(name, U, instrument) for U in metal]
    mfl = Stitch_spec(files)
    
    if phot:
        Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname,phot = True)  
    
    else:
        mfl = Resize(fit_wv, fit_flat, mwv, mfl)
        Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname)  

def Redden_and_fit2(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname, phot = False):    
    minidust = Gen_dust_minigrid(fit_wv,redshift)

    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
        redflgrid = mfl * dustgrid
        
        if phot:
            SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
            redflgrid = np.array([SCL]).T*redflgrid
            chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
            np.save(chi_path + '{0}_d{1}_{2}_SCL'.format(outname, i, instrument), np.array([SCL]).T)
        
        else:
            SCL = np.load(chi_path + '{0}_d{1}_phot_SCL.npy'.format(outname, i))
            redflgrid = SCL*redflgrid
            SCL2 = Scale_model_mult(fit_fl,fit_er,redflgrid)
            chigrid = np.sum(((fit_fl / np.array([SCL2]).T - redflgrid) / (fit_er / np.array([SCL2]).T)) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
        
        np.save(chi_path + '{0}_d{1}_{2}_chidata'.format(outname, i, instrument),chigrid)

####################################################################
####################################################################


class Gen_spec3(object):
    def __init__(self, field, galaxy_id, specz, g102_beam, g141_beam,
                 g102_lims = [7900, 11300], g141_lims = [11100, 16000],
                 filter_102 = 201, filter_141 = 203, tmp_err = False, 
                 phot_tmp_err = False, errterm = 0):
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
        
        field - GND/GSD/UDS
        galaxy_id - ID number from 3D-HST
        specz - z_grism
        g102_lims - window for g102
        g141_lims - window for g141
        tmp_err - (flag) whether or not we apply a template error function (not available)
        """
        self.Bwv, self.Bflx, self.Berr, self.Bflt = np.load(spec_path + '{0}_{1}_g102.npy'.format(field, galaxy_id))
        self.Rwv, self.Rflx, self.Rerr, self.Rflt = np.load(spec_path + '{0}_{1}_g141.npy'.format(field, galaxy_id))
        self.Bmask = np.load(spec_path + 'spec_mask/{0}_{1}_g102_mask.npy'.format(field, galaxy_id))
        self.Rmask = np.load(spec_path + 'spec_mask/{0}_{1}_g141_mask.npy'.format(field, galaxy_id))
        
        self.Bwv = self.Bwv[self.Bmask]
        self.Bflt = self.Bflt[self.Bmask]
        self.Bflx = self.Bflx[self.Bmask]
        self.Berr = self.Berr[self.Bmask] 
        
        self.Rwv = self.Rwv[self.Rmask]
        self.Rflt = self.Rflt[self.Rmask]
        self.Rflx = self.Rflx[self.Rmask]
        self.Rerr = self.Rerr[self.Rmask] 
        
        self.Pwv, self.Pflx, self.Perr, self.Pnum = np.load(phot_path + '{0}_{1}_phot.npy'.format(field, galaxy_id))
        self.Pwv_rf = self.Pwv / (1 + self.specz)
                
        self.IDB = [U for U in range(len(self.Bwv)) if g102_lims[0] <= self.Bwv[U] <= g102_lims[-1] and self.Bflx[U]**2 > 0]
        self.IDR = [U for U in range(len(self.Rwv)) if g141_lims[0] <= self.Rwv[U] <= g141_lims[-1] and self.Rflx[U]**2 > 0]

        self.Bwv = self.Bwv[self.IDB]
        self.Bwv_rf = self.Bwv / (1 + specz)
        self.Bflt = self.Bflt[self.IDB]
        self.Bflx = self.Bflx[self.IDB] #* Bscale
        self.Berr = self.Berr[self.IDB] #* Bscale
        
        self.Rwv = self.Rwv[self.IDR]
        self.Rwv_rf = self.Rwv / (1 + specz)
        self.Rflt = self.Rflt[self.IDR]
        self.Rflx = self.Rflx[self.IDR] #* Rscale
        self.Rerr = self.Rerr[self.IDR] #* Rscale

        self.model_photDF = pd.read_pickle(phot_path + 'model_photometry_list.pkl')
        
        self.IDP = []
        for i in range(len(self.Pnum)):
            for ii in range(len(self.model_photDF)):
                if self.Pnum[i] == self.model_photDF.tmp_num[self.model_photDF.index[ii]]:
                    self.IDP.append(ii)
        
        if phot_tmp_err:
            ewv, tmp= np.loadtxt(template_path + 'TEMPLATE_ERROR.eazy_v1.0').T
            iphterr = interp1d(ewv,tmp)(self.Pwv_rf)
            self.Perr_o = self.Perr
            self.Perr = np.sqrt(self.Perr**2 + (iphterr * self.Pflx)**2+ (errterm * self.Pflx)**2)
            
#         if tmp_err:
#             WV,TEF = np.load(data_path + 'template_error_function.npy')
#             iTEF = interp1d(WV,TEF)(self.gal_wv_rf)
#             self.gal_er = np.sqrt(self.gal_er**2 + (iTEF*self.fl)**2)

        self.Bbeam = model.BeamCutout(fits_file = g102_beam)
        self.Rbeam = model.BeamCutout(fits_file = g141_beam)

        #self.Bpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g102.A.fits')
        #self.Rpoint_beam = model.BeamCutout(fits_file = '../beams/point_41086.g141.A.fits')
        
        ### Define precalculated terms for photometry
        self.sens_wv, self.trans = np.load(template_path + 'master_tmp.npy')
        self.b = np.load(template_path + 'bottom_precalc.npy')
        self.dnu = np.load(template_path + 'dnu_precalc.npy')
        self.adj = np.load(template_path + 'adj_precalc.npy')
        self.mdleffwv = np.load(template_path + 'effwv_precalc.npy') 
    
        ### Set transmission curve
        model_wave, model_flux = np.load(model_path + 'm0.019_a2.0_dt0_spec.npy')
        Bw,Bfl = self.Sim_spec_indv(self.Bbeam, model_wave, np.ones(len(model_wave)))
        Rw,Rfl = self.Sim_spec_indv(self.Rbeam, model_wave, np.ones(len(model_wave)))
        self.Btrans = interp1d(Bw,Bfl)(self.Bwv)       
        self.Rtrans = interp1d(Rw,Rfl)(self.Rwv) 
    
    def Sim_spec_indv(self, BEAM, model_wave, model_flux):
        ### creates a model using an individual beam
        BEAM.beam.compute_model(spectrum_1d=[model_wave, model_flux])
        w, f, e = BEAM.beam.optimal_extract(BEAM.beam.model, bin=0)
        return w, f
        
    def Sim_spec_mult(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bbeam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rbeam, model_wave, model_flux)

    def Sim_spec_mult_point(self, model_wave, model_flux):
        ### creates a model for g102 and g141 using individual beams
        return self.Sim_spec_indv(self.Bpoint_beam, model_wave, model_flux), \
                self.Sim_spec_indv(self.Rpoint_beam, model_wave, model_flux)
    
    def Sim_spec(self, metal, age, tau, model_redshift = 0, Av = 0, multi_component = False,
                point_scale=1):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))

        [Bmw, Bmf], [Rmw, Rmf] = self.Sim_spec_mult(model_wave * (1 + model_redshift), 
                                                                        model_flux * Calzetti(Av,model_wave))
        iBmf = interp1d(Bmw,Bmf)(self.Bwv)       
        iRmf = interp1d(Rmw,Rmf)(self.Rwv)     
        
        self.Bmfl = iBmf / self.Btrans
        self.Rmfl = iRmf / self.Rtrans
            
        self.Bmfl *= self.PC
        self.Rmfl *= self.PC
        
        if multi_component:
            [Bpmw, Bpmf], [Rpmw, Rpmf] = self.Sim_spec_mult_point(model_wave * (1 + model_redshift), 
                                                                            model_flux * Calzetti(Av,model_wave))
            iBpmf = interp1d(Bpmw,Bpmf)(self.Bwv)       
            iRpmf = interp1d(Rpmw,Rpmf)(self.Rwv)     

            self.Bpmfl = iBpmf / self.Bflt
            self.Rpmfl = iRpmf / self.Rflt

            self.Bpmfl *= Scale_model(self.Bflx, self.Berr, self.Bpmfl)
            self.Rpmfl *= Scale_model(self.Rflx, self.Rerr, self.Rpmfl)
            
            self.Bpmfl *= point_scale
            self.Rpmfl *= point_scale
            
            self.BMCmfl = self.Bmfl + self.Bpmfl
            self.RMCmfl = self.Rmfl + self.Rpmfl
            
            self.BMCmfl *= Scale_model(self.Bflx, self.Berr, self.BMCmfl)
            self.RMCmfl *= Scale_model(self.Rflx, self.Rerr, self.RMCmfl)
       
    def Sim_phot_mult(self, model_wave, model_flux):
        
        imfl =interp1d(self.c / model_wave, (self.c/(self.c / model_wave)**2) * model_flux)

        mphot = (np.trapz(imfl(self.c /(self.sens_wv[self.IDP])).reshape([len(self.IDP),len(self.sens_wv[0])]) \
                          * self.b[self.IDP],self.dnu[self.IDP])/np.trapz(self.b[self.IDP],
                                                                          self.dnu[self.IDP])) * self.adj[self.IDP]
        
        return np.array([self.mdleffwv[self.IDP],mphot])

    def Sim_phot(self, metal, age, tau, model_redshift = 0, Av = 0):
        if model_redshift ==0:
            model_redshift = self.specz

        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(
            metal, age, tau))
        
        self.Pmwv, self.Pmfl = self.Sim_phot_mult(model_wave * (1 + model_redshift), 
                                                  model_flux * Calzetti(Av,model_wave))
        self.PC =  Scale_model(self.Pflx, self.Perr, self.Pmfl)  
        self.Pmfl = self.Pmfl * self.PC
        
    def Sim_all(self, metal, age, tau, model_redshift = 0, Av = 0):
        self.Sim_phot(metal, age, tau, model_redshift, Av)
        self.Sim_spec(metal, age, tau, model_redshift, Av)
        
    def Scale_flux(self, bfZ, bft, bftau, bfz, bfd):
        model_wave, model_flux = np.load(model_path + 'm{0}_a{1}_dt{2}_spec.npy'.format(bfZ, bft, bftau))
        
        US_model_flux = F_lam_per_M(model_flux * Calzetti(bfd,model_wave), model_wave * (1 + bfz), bfz, 0, 1)

        US_pwv, US_pfl = self.Sim_phot_mult(model_wave * (1 + bfz), US_model_flux)
        
        self.mass = Scale_model(self.Pflx, self.Perr, US_pfl)
        
        self.lmass = np.log10(self.mass)
        
        self.model_wave = model_wave
        self.S_model_flux = US_model_flux * self.mass
          
        Bw,Bf = self.Sim_spec_indv(self.Bbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        Rw,Rf = self.Sim_spec_indv(self.Rbeam, self.model_wave * (1 + bfz), self.S_model_flux)
        
        iBmf = interp1d(Bw,Bf)(self.Bwv)       
        iRmf = interp1d(Rw,Rf)(self.Rwv)  
  
        Bmfl = iBmf / self.Btrans
        Rmfl = iRmf /self.Rtrans

        self.Bscale = Scale_model(self.Bflx, self.Berr, Bmfl)
        self.Rscale = Scale_model(self.Rflx, self.Rerr, Rmfl)
        
        self.Bflx = self.Bflx / self.Bscale ; self.Berr = self.Berr / self.Bscale 
        self.Rflx = self.Rflx / self.Rscale ; self.Rerr = self.Rerr / self.Rscale 
        
        
def Fit_all3(field, galaxy, g102_beam, g141_beam, specz, metal, age, tau, rshift, dust, name, bfZ, bft, bftau, bfz, bfd,
        gen_models = True, age_conv= data_path + 'light_weight_scaling_3.npy', errterm = 0,
           outname = 'none'):
   
    if outname == 'none':
        outname = name
    ######## initialize spec
    sp = Gen_spec3(field, galaxy, specz, g102_beam, g141_beam, phot_tmp_err = True, errterm = errterm)
    
    sp.Scale_flux(bfZ, bft, bftau, bfz, bfd)
    
    if gen_models:
        Gen_mflgrid(sp, name, metal, age, tau, rshift)

    ## set some variables
    wv,fl = np.load(model_path + 'm0.019_a2.0_dt8.0_spec.npy')
    [Bmwv,Bmflx], [Rmwv,Rmflx] = sp.Sim_spec_mult(wv,fl)
    
    Stitch_resize_redden_fit3(sp.Pwv, sp.Pflx, sp.Perr, 'none', 'phot', name, sp.Pwv, 
                     metal, age, tau, rshift, outname, phot = True) 
    
    Stitch_resize_redden_fit3(sp.Bwv, sp.Bflx, sp.Berr, sp.Btrans, 'g102', name, Bmwv, 
                     metal, age, tau, rshift, outname)
    Stitch_resize_redden_fit3(sp.Rwv, sp.Rflx, sp.Rerr, sp.Rtrans, 'g141', name, Rmwv, 
                     metal, age, tau, rshift, outname)

    
    P, PZ, Pt, Ptau, Pz, Pd = Analyze_full_fit(outname, metal, age, tau, rshift, 
                                               dust=dust,age_conv = age_conv)

    np.save(out_path + '{0}_tZ_pos'.format(outname),P)
    np.save(out_path + '{0}_Z_pos'.format(outname),[metal,PZ])
    np.save(out_path + '{0}_t_pos'.format(outname),[age,Pt])
    np.save(out_path + '{0}_tau_pos'.format(outname),[np.append(0, np.power(10, np.array(tau)[1:] - 9)),Ptau])
    np.save(out_path + '{0}_rs_pos'.format(outname),[rshift,Pz])
    np.save(out_path + '{0}_d_pos'.format(outname),[dust,Pd])


def Stitch_resize_redden_fit3(fit_wv, fit_fl, fit_er, fit_flat, instrument, name, mwv, 
                     metal, age, tau, redshift, outname, phot=False):
    #############Read in spectra and stich spectra grid together#################
    files = [chi_path + 'spec_files/{0}_m{1}_{2}.npy'.format(name, U, instrument) for U in metal]
    mfl = Stitch_spec(files)
    
    if phot:
        Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname,phot = True)  
    
    else:
        mfl = Resize(fit_wv, fit_flat, mwv, mfl)
        Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname)  

def Redden_and_fit3(fit_wv, fit_fl, fit_er, mfl, metal, age, tau, redshift, instrument, name, outname, phot = False):    
    minidust = Gen_dust_minigrid(fit_wv,redshift)

    Av = np.round(np.arange(0, 1.1, 0.1),1)
    for i in range(len(Av)):
        dustgrid = np.repeat([minidust[str(Av[i])]], len(metal)*len(age)*len(tau), axis=0).reshape(
            [len(minidust[str(Av[i])])*len(metal)*len(age)*len(tau), len(fit_wv)])
        redflgrid = mfl * dustgrid
        
        if phot:
            SCL = Scale_model_mult(fit_fl,fit_er,redflgrid)
            redflgrid = np.array([SCL]).T*redflgrid
            chigrid = np.sum(((fit_fl - redflgrid) / fit_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
            np.save(chi_path + '{0}_d{1}_{2}_SCL'.format(outname, i, instrument), np.array([SCL]).T)
        
        else:
            SCL = np.load(chi_path + '{0}_d{1}_phot_SCL.npy'.format(outname, i))
            chigrid = np.sum(((fit_fl - redflgrid * SCL) / (fit_er)) ** 2, axis=1).reshape([len(metal), len(age), len(tau), len(redshift)])
        
        np.save(chi_path + '{0}_d{1}_{2}_chidata'.format(outname, i, instrument),chigrid)