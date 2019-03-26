#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_exam import Gen_spec
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
import fsps
import dynesty
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy import stats
from sim_engine import forward_model_grism, Salmon
from spec_id import Scale_model
from spec_tools import Oldest_galaxy
from spec_stats import Get_posterior
from astropy.cosmology import Planck13 as cosmo
from multiprocessing import Pool
from prospect.models.transforms import logsfr_ratios_to_masses

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
    
if __name__ == '__main__':
    runnum = sys.argv[1] 
    rndseed = int(sys.argv[2])

#####SET SIM#####
specz = 1.25

sim2 = Gen_spec('GND', 21156, 1.25257,
               g102_lims=[8300, 11288], g141_lims=[11288, 16500],mdl_err = False,
            phot_errterm = 0.0, decontam = False) 

sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(0.8), sfh = 3, dust_type = 1)
sp.params['dust2'] =0.2
sp.params['dust1'] =0.2

tab_sfh = np.array([0.9, 0.3, 0.025, 0.001, 0.0001, 0.001, 0.00001, 0.0002, 0.002, 0.0001])

#######################
#######set LBT#########
lages = [0,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9]

tuniv = Oldest_galaxy(specz)
nbins = len(lages) - 1

tbinmax = (tuniv * 0.85) * 1e9
lim1, lim2 = 7.4772, 8.0
agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
agebins = np.array([agelims[:-1], agelims[1:]]).T

LBT = (10**agebins.T[1][::-1][0] - 10**agebins.T[0][::-1])*1E-9
time_per_bin = np.diff(10**agebins, axis=-1)[:,0]

#########################

sp.set_tabular_sfh(LBT,tab_sfh)

wave1, flux1 = sp.get_spectrum(tage = 4.25, peraa = True)

mass_perc1 = sp.stellar_mass
 
D_l = cosmo.luminosity_distance(specz).value # in Mpc
conv = 3.086E24
lsol_to_fsol = 3.839E33

mass_transform = (10**11 / mass_perc1) * lsol_to_fsol / (4 * np.pi * (D_l*conv)**2)
    
sim2.Make_sim(wave1, flux1 * mass_transform, specz, perturb = False)

#####RESET FSPS#####
sp = fsps.StellarPopulation(imf_type = 2, tpagb_norm_type=0, zcontinuous = 1, logzsol = np.log10(1), sfh = 3, dust_type = 1)

#######################
#######reset LBT#########
lages = [0,8.0,8.3,8.6,9.0,9.3,9.6,10]

tuniv = Oldest_galaxy(specz)
nbins = len(lages) - 1

tbinmax = (tuniv * 0.85) * 1e9
lim1, lim2 = 7.4772, 8.0
agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
agebins = np.array([agelims[:-1], agelims[1:]]).T

LBT = (10**agebins.T[1][::-1][0] - 10**agebins.T[0][::-1])*1E-9
time_per_bin = np.diff(10**agebins, axis=-1)[:,0]

############
###priors###
agelim = Oldest_galaxy(specz)

def tab_prior(u):
    m = (0.03 * u[0] + 0.001) / 0.019
    
    a = (agelim - LBT[0])* u[1] + LBT[0]
    
    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8]])

    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)

    masses = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = agebins) * 1E9

    t1, t2, t3, t4, t5, t6, t7 = np.array(masses / time_per_bin)[::-1]
    
    z = stats.norm.ppf(u[9],loc = specz, scale = 0.003)
    
    d = u[10]
    
    lm = stats.norm.ppf(u[11],loc = 10.75, scale = 0.5)
    
    return [m, a, t1, t2, t3, t4, t5, t6, t7, z, d, lm]

############
#likelihood#

def Gather_grism_sim_data(spec):
    wvs = []
    flxs = []
    errs = []
    beams = []
    trans = []
    
    if spec.g102:
        wvs.append(spec.Bwv)
        flxs.append(spec.SBfl)
        errs.append(spec.SBer)
        beams.append(spec.Bbeam)
        trans.append(spec.Btrans)
    
    if spec.g141:
        wvs.append(spec.Rwv)
        flxs.append(spec.SRfl)
        errs.append(spec.SRer)
        beams.append(spec.Rbeam)
        trans.append(spec.Rtrans)

    return np.array([wvs, flxs, errs, beams, trans])

def forward_model_all_beams(beams, trans, in_wv, model_wave, model_flux):
    FL = np.zeros([len(beams),len(in_wv)])

    for i in range(len(beams)):
        mwv, mflx = forward_model_grism(beams[i], model_wave, model_flux)
        FL[i] = interp1d(mwv, mflx)(in_wv)
        FL[i] /= trans[i]

    return np.mean(FL.T,axis=1)

def Full_forward_model(spec, wave, flux, specz):
    Gmfl = []
    
    for i in range(len(wvs2)):
        Gmfl.append(forward_model_all_beams(beams2[i], trans2[i], wvs2[i], wave * (1 + specz), flux))

    Pmfl = spec.Sim_phot_mult(wave * (1 + specz),flux)

    return np.array(Gmfl), Pmfl


def Full_fit(spec, Gmfl, Pmfl):
    Gchi = 0
    
    for i in range(len(wvs2)):
        #scale = Scale_model(flxs2[i], errs2[i], Gmfl[i])
        #Gchi = Gchi + np.sum(((((flxs2[i] / scale) - Gmfl[i]) / (errs2[i] / scale))**2))
        Gchi = Gchi + np.sum( ((flxs2[i] - Gmfl[i]) / errs2[i])**2 )
    Pchi = np.sum((((spec.SPflx - Pmfl) / spec.SPerr)**2))
    
    return Gchi, Pchi

wvs2, flxs2, errs2, beams2, trans2 = Gather_grism_sim_data(sim2)

def tab_L(X):
    m, a, t1, t2, t3, t4, t5, t6, t7, z, d, lm = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)

    sp.set_tabular_sfh(LBT,np.array([t1, t2, t3, t4, t5, t6, t7]))
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)
    
    D_l = cosmo.luminosity_distance(z).value # in Mpc

    mass_transform = (10**lm / sp.stellar_mass) * lsol_to_fsol / (4 * np.pi * (D_l*conv)**2)  

    Gmfl, Pmfl = Full_forward_model(sim2, wave, flux * mass_transform, z)
      
    Gchi, Pchi = Full_fit(sim2, Gmfl, Pmfl)
                  
    return -0.5 * (Gchi + Pchi)

############
####run#####
d_tsampler = dynesty.DynamicNestedSampler(tab_L, tab_prior, ndim = 12, sample = 'rwalk', bound = 'multi',
                                  queue_size = 8, pool = Pool(processes=8))  
d_tsampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=False)

dres = d_tsampler.results
############
####save####
np.save(out_path + 'sim_test_tab_to_tab_lessbin_{0}'.format(runnum), dres) 

sp.params['compute_light_ages'] = True
 
lwa = []

for ii in range(len(dres.samples)):
    bfZ, bft, bftau1, bftau2, bftau3, bftau4, bftau5, bftau6, bftau7, bfz, bfd, bfm = dres.samples[-1]

    sp.params['dust2'] = bfd
    sp.params['dust1'] = bfd
    sp.params['logzsol'] = np.log10(bfZ)

    sp.set_tabular_sfh(LBT,np.array([bftau1, bftau2, bftau3, bftau4, bftau5, bftau6, bftau7]))

    lwa.append(sp.get_mags(tage = bft, bands =['sdss_g'])[0])
       
sp.params['compute_light_ages'] = False

np.save(out_path + 'sim_test_tab_to_tab_lessbin_{0}_lwa'.format(runnum), lwa) 

params = ['m', 'a','t1', 't2', 't3', 't4', 't5', 't6', 't7', 'z', 'd', 'lm']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + 'sim_test_tab_to_tab_lessbin_{0}_P{1}'.format(runnum, params[i]),[t,pt])

bfm, bfa, bft1, bft2, bft3, bft4, bft5, bft6, bft7, bfz, bfd, bflm = dres.samples[-1]

np.save(pos_path + 'sim_test_tab_to_tab_lessbin_{0}_bfit'.format(runnum),
        [bfm, bfa, bft1, bft2, bft3, bft4,bft5, bft6, bft7, bfz, bfd, bflm, dres.logl[-1]])
    
dres.samples[:,1] = lwa
m,Pm = Get_posterior(dres, 1)
np.save(pos_path + 'sim_test_tab_to_tab_lessbin_{0}_Plwa'.format(runnum),[m,Pm])




