__author__ = 'vestrada'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
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
from spec_exam import Gen_ALMA_spec
import sys
import fsps
import dynesty
from scipy.interpolate import interp1d
from scipy import stats
from spec_tools import Oldest_galaxy
from astropy.cosmology import Planck13 as cosmo
from multiprocessing import Pool
from prospect.models.transforms import logsfr_ratios_to_masses
from spec_stats import Get_posterior

hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/fdata/scratch/vestrada78840/chidat/'
    pos_path = '/home/vestrada78840/posteriors/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/out_dict/'
    pos_path = '../data/posteriors/'
    phot_path = '../phot/'

##############################
###set useful distributions###
##############################

def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    C = np.trapz(G, x)
    G /= C
    return G

rZ = np.arange(0.0019, 0.0302,0.0001)
r1 = np.arange(0.4999,1.5002,0.0001)
r2 = np.arange(-0.5001,0.5002,0.0001)

gZ= Gauss_dist(rZ,0.019,0.005)
g1 = Gauss_dist(r1,1,0.25)
g2 = Gauss_dist(r2,0,0.25)

iCZ = interp1d(np.cumsum(gZ) / np.cumsum(gZ).max(), rZ,fill_value=1, bounds_error=False)
iC1 = interp1d(np.cumsum(g1) / np.cumsum(g1).max(), r1,fill_value=1, bounds_error=False)
iC2 = interp1d(np.cumsum(g2) / np.cumsum(g2).max(), r2,fill_value=0, bounds_error=False)


#################################
###functiongs needed for prior###
#################################

def convert_sfh(agebins, mformed, epsilon=1e-4, maxage=None):
    #### create time vector
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
    bin_edges = np.unique(agebins_yrs)
    if maxage is None:
        maxage = agebins_yrs.max()  # can replace maxage with something else, e.g. tuniv
    t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
    t.sort()
    t = t[1:-1] # remove older than oldest bin, younger than youngest bin
    fsps_time = maxage - t

    #### calculate SFR at each t
    sfr = mformed / dt
    sfrout = np.zeros_like(t)
    sfrout[::2] = sfr
    sfrout[1::2] = sfr  # * (1+epsilon)

    return (fsps_time / 1e9)[::-1], sfrout[::-1], maxage / 1e9

def get_lwa(params, agebins):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_g'])
    sp.params['compute_light_ages'] = False
    
    return lwa
    
####### may change to be agelim
def get_agebins(maxage):
    lages = [0,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9]
    
    nbins = len(lages) - 1

    tbinmax = (maxage * 0.85) * 1e9
    lim1, lim2 = 7.4772, 8.0
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(maxage*1e9)]
    return np.array([agelims[:-1], agelims[1:]]).T



#############
####prior####
#############

def galfit_prior(u):
#     m = (0.03*u[0] + 0.001) / 0.019
    m = iCZ(u[0]) / 0.019
    
    a = (agelim - 1)* u[1] + 1
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9], u[10], u[11]])

    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)

    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a)) * 1E9
    
    z = stats.norm.ppf(u[12],loc = specz, scale = 0.005)
    
    d = u[13]
    
    bp1 = iC2(u[14])
    
    rp1 = iC2(u[15])
    
    lwa = get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a))
        
    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, bp1, rp1, lwa]

def zfit_prior(u):
    m = iCZ(u[0]) / 0.019
    
    a = (2)* u[1] + 1
    
    z = 2.5 * u[2]
        
    return [m, a, z]


##########################
#functions for likelihood#
##########################

def forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def Full_scale(spec, Pmfl):
    return Scale_model(spec.Pflx, spec.Perr, Pmfl)

def Gather_grism_data(spec):
    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    
    if spec.g102:
        wvs.append(spec.Bwv)
        flxs.append(spec.Bfl)
        errs.append(spec.Ber)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.Rfl)
        errs.append(spec.Rer)
        beams.append(spec.Rbeam)
        trans.append(spec.Rtrans)

    return np.array([wvs, flxs, errs, beams, trans])

def Full_forward_model(spec, wave, flux, specz, wvs, flxs, errs, beams, trans):
    Gmfl = []
    
    for i in range(len(wvs)):
        Gmfl.append(forward_model_all_beams(beams[i], trans[i], wvs[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl

def Full_calibrate(Gmfl, p1, wvs):
    for i in range(len(wvs)):
        Gmfl[i] = Gmfl[i] * ((p1[i] * wvs[i]) / (wvs[i][-1] - wvs[i][0]) + 5)
    return Gmfl

def Calibrate_grism(spec, Gmfl, p1, wvs, flxs, errs):
    linecal = []
    for i in range(len(wvs)):
        lines = ((p1[i] * wvs[i]) / (wvs[i][-1] - wvs[i][0]) + 5)
        scale = Scale_model(flxs[i]  / lines, errs[i] / lines, Gmfl[i])    
        linecal.append(scale * lines)
        
    return linecal


def Full_fit(spec, Gmfl, Pmfl, wvs, flxs, errs):
    Gchi = 0
    
    for i in range(len(wvs)):
        scale = Scale_model(flxs[i], errs[i], Gmfl[i])
        Gchi = Gchi + np.sum(((((flxs[i] / scale) - Gmfl[i]) / (errs[i] / scale))**2))
    
    Pchi = np.sum((((spec.Pflx - Pmfl) / spec.Perr)**2))
    
    return Gchi, Pchi

#####################
#####Likelihoods#####
#####################

def galfit_L(X):
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, bp1, rp1, lwa = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr)    
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z)
       
    Gmfl = Full_calibrate(Gmfl, [bp1, rp1])
        
    PC= Full_scale(Gs, Pmfl)

    Gchi, Pchi = Full_fit(Gs, Gmfl, PC*Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

def zfit_L(X):
    m, a, z = X

    sp.params['logzsol'] = np.log10(m)
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, flux, z)
              
    PC= Full_scale(Gs, Pmfl)

    Gchi, Pchi = Full_fit(Gs, Gmfl, PC*Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

#######################
###Fitting functions###
#######################

class zfit(object):
    def __init__(self, field, galaxy, poolsize = 8, verbose = False):
        #########define fsps#########
        self.sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1),sfh = 0)

        ###########gen spec##########
        self.Gs = Gen_spec(field, galaxy, 1, g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = False,
                phot_errterm = 0.02, irac_err = 0.04, decontam = True) 

        ####generate grism items#####
        self.wvs, self.flxs, self.errs, self.beams, self.trans = Gather_grism_data(self.Gs)

        #######set up dynesty########
        sampler = dynesty.NestedSampler(self.zfit_L, self.zfit_prior, ndim = 3, sample = 'rwalk', bound = 'multi',
                                            pool=Pool(processes= poolsize), queue_size = poolsize)

        sampler.run_nested(print_progress = verbose)

        dres = sampler.results

        np.save(out_path + '{0}_{1}_zfit'.format(field, galaxy), dres) 

        ##save out P(z) and bestfit##

        t,pt = Get_posterior(dres,2)
        np.save(pos_path + '{0}_{1}_zfit_Pz'.format(field, galaxy),[t,pt])

        bfm, bfa, bfz = dres.samples[-1]

        np.save(pos_path + '{0}_{1}_zfit_bfit'.format(field, galaxy), [bfm, bfa,bfz, dres.logl[-1]])
        
    def zfit_L(self,X):
        m, a, z = X

        self.sp.params['logzsol'] = np.log10(m)

        wave, flux = self.sp.get_spectrum(tage = a, peraa = True)

        Gmfl, Pmfl = Full_forward_model(self.Gs, wave, flux, z, self.wvs, self.flxs, self.errs, self.beams, self.trans)

        PC= Full_scale(self.Gs, Pmfl)

        Gchi, Pchi = Full_fit(self.Gs, Gmfl, PC*Pmfl, self.wvs, self.flxs, self.errs)

        return -0.5 * (Gchi + Pchi)
        
    def zfit_prior(self,u):
        m = iCZ(u[0]) / 0.019

        a = (2)* u[1] + 1

        z = 2.5 * u[2]

        return [m, a, z]