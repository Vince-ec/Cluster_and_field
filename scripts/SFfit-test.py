#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'
    
verbose=False
poolsize = 8


class Gen_SF_spec_2(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8200, 11300], g141_lims=[11200, 16000],
                phot_errterm = 0, irac_err = None, mask = True):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims
        self.g102_beam = glob(cbeam_path + '*{0}*g102*'.format(galaxy_id))
        self.g141_beam = glob(cbeam_path + '*{0}*g141*'.format(galaxy_id))
        self.sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = np.log10(1), sfh = 4, tau = 0.1, dust_type = 1)
        self.mask = mask

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
        self.Bwv, self.Bwv_rf, self.Bflx, self.Berr, self.Bflt, self.IDB, self.Bline, self.Bcont = load_spec_SF_2(self.field,
                            self.galaxy_id, 'g102', self.g102_lims,  self.specz, mask = self.mask)

        self.Bfl = self.Bflx / self.Bflt 
        self.Bbeam, self.Btrans = load_beams_and_trns(self.Bwv, self.g102_beam)
        self.Ber = self.Berr / self.Bflt
        self.g102 = True
        
        self.Rwv, self.Rwv_rf, self.Rflx, self.Rerr, self.Rflt, self.IDR, self.Rline, self.Rcont = load_spec_SF_2(self.field,
                            self.galaxy_id, 'g141', self.g141_lims,  self.specz, mask = self.mask)

        self.Rfl = self.Rflx / self.Rflt 
        self.Rbeam, self.Rtrans = load_beams_and_trns(self.Rwv, self.g141_beam)
        self.Rer = self.Rerr / self.Rflt
        self.g141 = True

        self.Pwv, self.Pwv_rf, self.Pflx, self.Perr, self.Pnum = load_spec_SF_2(self.field,
                                self.galaxy_id, 'phot', self.g141_lims,  self.specz, grism = False)
         
        self.Perr = apply_phot_err(self.Pflx, self.Perr, self.Pnum, base_err = phot_errterm, irac_err = irac_err)
        # load photmetry precalculated values
        self.model_photDF, self.IDP, self.sens_wv, self.trans, self.b, self.dnu, self.adj, self.mdleffwv = load_phot_precalc(self.Pnum)
    
    def Sim_spec_premade(self, model_wave, model_flux):
        if self.g102:
            self.Bmfl = self.Forward_model_all_beams_flatted(self.Bbeam, self.Btrans, self.Bwv, model_wave, model_flux)
            self.Bmfl *= self.PC

            if not self.set_scale:
                Bscale = Scale_model(self.Bfl, self.Ber, self.Bmfl)

                self.Bfl = self.Bfl / Bscale ; self.Ber = self.Ber / Bscale 
                
        if self.g141: 
            self.Rmfl = self.Forward_model_all_beams_flatted(self.Rbeam, self.Rtrans, self.Rwv, model_wave, model_flux) 
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
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
            
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)
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
            self.Bfl =  self.Bfl/ self.bcal/ self.bscale
            self.Ber =  self.Ber/ self.bcal/ self.bscale
               
        if self.g141:
            self.rcal = Calibrate_grism([self.Rwv, self.Rfl, self.Rer], self.Rmfl, rp1)
            self.Rfl =  self.Rfl/ self.rcal/ self.rscale
            self.Rer =  self.Rer/ self.rcal/ self.rscale
        
def load_spec_SF_2(field, galaxy_id, instr, lims, specz, grism = True, mask = True):
    # if loading photometry FLT stands in for num
    bfilters = [34, 36, 37, 58, 117, 118, 195, 196, 220, 224]

    if grism:
        W, F, E, FLT, L, C = np.load('../CLEAR_show_and_tell/{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))
        
        IDX = [U for U in range(len(W)) if lims[0] <= W[U] <= lims[-1] and F[U]**2 > 0]

        W = np.array(W[IDX])
        WRF = np.array(W / (1 + specz))
        FLT = np.array(FLT[IDX])
        F = np.array(F[IDX]) 
        E = np.array(E[IDX]) 
        L = np.array(L[IDX]) 
        C = np.array(C[IDX]) 
        
        if mask:
            IDT = get_mask(field, galaxy_id, W, instr)
            return W[IDT], WRF[IDT], F[IDT], E[IDT], FLT[IDT], np.array(IDX)[IDT], L[IDT], C[IDT]
        
        else:
            return W, WRF, F, E, FLT, np.array(IDX), L, C

    else:
        W, F, E, FLT = np.load('../CLEAR_show_and_tell/{0}_{1}_{2}.npy'.format(field, galaxy_id, instr))
        
        WRF = W / (1 + specz)
        
        IDX = []
        
        for i in range(len(FLT)):
            if FLT[i] not in bfilters and F[i] / E[i] > 0.5:
                IDX.append(i)
        
        W, WRF, F, E, FLT = W[IDX], WRF[IDX], F[IDX], E[IDX], FLT[IDX]
        
        W, WRF, F, E, FLT = W[F > 0], WRF[F > 0], F[F > 0], E[F > 0], FLT[F > 0]
        
        return W, WRF, F, E, FLT


def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (2)* u[1] + 0.1
        
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a, binnum = 6))
    
    lm = Gaussian_prior(u[8], [9.5, 11.5], 11, 0.75)
     
    d = 4*u[9]
    
    bp1 = Gaussian_prior(u[10], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[11], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[12], [0.1,10])
    bb = log_10_prior(u[13], [0.0001,1])
    bl = log_10_prior(u[14], [0.01,1])
    
    ra = log_10_prior(u[15], [0.1,10])
    rb = log_10_prior(u[16], [0.0001,1])
    rl = log_10_prior(u[17], [0.01,1])
     
    glz = Gaussian_prior(u[18], [-2,2], 0.0, 1)
        
    return [m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl, glz]

def Galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl,glz = X
    
    sp.params['dust2'] = d
    sp.params['logzsol'] = np.log10(m)
    sp.params['gas_logz'] = glz

    time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**lm, specz, 
                                    wvs, flxs, errs, beams, trans)
       
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
sp.params['add_neb_emission']=1
sp.params['gas_logz'] = -1.0
sp.params['gas_logu'] = -2

###########gen spec##########
Gs = Gen_SF_spec_2('GND', 37738, 1.4507, mask = False)
specz = 1.4507
####generate grism items#####
wvs, flxs, errs, beams, trans = Gather_grism_data(Gs)

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 19, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=True)

#sampler = dynesty.NestedSampler(Galfit_L, Galfit_prior, ndim = 19, nlive_points = 4000,
#                                         sample = 'rwalk', bound = 'multi',
#                                         pool=Pool(processes=8), queue_size=8)

#sampler.run_nested(print_progress=True)

dres = sampler.results

np.save(out_path + '{0}_{1}_SFfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'lm', 'd', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_SFfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bgu = dres.samples[-1]

np.save(pos_path + '{0}_{1}_SFfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bflm, bfd, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bgu, dres.logl[-1]])