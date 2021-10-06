#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    
verbose=False
poolsize = 8

agelim = Oldest_galaxy(specz)
#zscale = 0.035 * (1 + specz)
zscale = 0.0035 * (1 + specz)

def Galfit_prior(u):
    m = Gaussian_prior(u[0], [0.002,0.03], 0.019, 0.08)/ 0.019
    a = (agelim - 1)* u[1] + 1

    tsamp = np.array([u[2],u[3],u[4],u[5],u[6],u[7],u[8], u[9], u[10],u[11]])
    taus = stats.t.ppf( q = tsamp, loc = 0, scale = 0.3, df =2.)
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = logsfr_ratios_to_masses(logmass = 0, logsfr_ratios = taus, agebins = get_agebins(a))
    
    lm = Gaussian_prior(u[12], [9.5, 12.5], 11, 0.75)
  
    z = stats.norm.ppf(u[13],loc = specz, scale = zscale)
    
    d = log_10_prior(u[14],[1E-3,2])
    
    #bsc= Gaussian_prior(u[15], [0.8, 1.2], 1, 0.05)
    #rsc= Gaussian_prior(u[16], [0.8, 1.2], 1, 0.05)
    bp1 = Gaussian_prior(u[15], [-0.1,0.1], 0, 0.05)
    rp1 = Gaussian_prior(u[16], [-0.05,0.05], 0, 0.025)
    
    ba = log_10_prior(u[17], [0.1,10])
    bb = log_10_prior(u[18], [0.0001,1])
    bl = log_10_prior(u[19], [0.01,1])
    
    ra = log_10_prior(u[20], [0.1,10])
    rb = log_10_prior(u[21], [0.0001,1])
    rl = log_10_prior(u[22], [0.01,1])
   
    lwa = get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a),sp)[0]
    
    #return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bsc, rsc, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa]
    return [m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa]

def Galfit_L(X):
    #m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bsc, rsc, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa = X
    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d, bp1, rp1, ba, bb, bl, ra, rb, rl, lwa = X
    
    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 
    
    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Gmfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1+z),z,0,sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
       
    #Gmfl = Full_calibrate(Gmfl, [bp1, rp1], [bsc, rsc], wvs)
    Gmfl = Full_calibrate_2(Gmfl, [bp1, rp1], wvs, flxs, errs)
   
    return Full_fit_2(Gs, Gmfl, Pmfl, [ba,ra], [bb,rb], [bl, rl], wvs, flxs, errs)

############ simulate data with no offsets#############

def Q_spec_sim(Gs, bestfits):
    wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

    m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lm, z, d,\
        bp1, rp1, ba, bb, bl, ra, rb, rl = BFS

    sp.params['dust2'] = d
    sp.params['dust1'] = d
    sp.params['logzsol'] = np.log10(m)
    
    time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = a, peraa = True)

    Smfl, Pmfl = Full_forward_model(Gs, wave, F_lam_per_M(flux,wave*(1 + z), z, 0, sp.stellar_mass)*10**lm, z, 
                                    wvs, flxs, errs, beams, trans)
    
    return wvs, Smfl, errs, beams, trans, Pmfl,  

#########define fsps#########
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

###########gen spec##########
Gs = Gen_spec(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 
print(Gs.g102)
print(Gs.g141)

####generate grism items#####
#full_db = pd.read_pickle(data_path + 'all_galaxies_1d.pkl')
full_db = pd.read_pickle('../dataframes/fitdb/all_galaxies_1d.pkl')

params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm', 'z', 'd', 'bp1', 
          'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl']
BFS = []

for i in params:
    BFS.append(sfdb.query('field == "{}" and id == {}'.format(field,galaxy))['bf{}'.format(i)].values[0])

wvs, flxs, errs, beams, trans, Spmfl = Q_spec_sim(Gs, BFS)

Gs.Pflx = Spmfl

#######set up dynesty########
sampler = dynesty.DynamicNestedSampler(Galfit_L, Galfit_prior, ndim = 24, nlive_points = 4000,
                                         sample = 'rwalk', bound = 'multi',
                                         pool=Pool(processes=8), queue_size=8)

sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, print_progress=False)

dres = sampler.results

np.save(out_path + '{0}_{1}_tabfit'.format(field, galaxy), dres) 

##save out P(z) and bestfit##

#params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm',
#          'z', 'd', 'bsc', 'rsc', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']
params = ['m', 'a', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'lm',
          'z', 'd', 'bp1', 'rp1', 'ba', 'bb', 'bl', 'ra', 'rb', 'rl', 'lwa']
for i in range(len(params)):
    t,pt = Get_posterior(dres,i)
    np.save(pos_path + '{0}_{1}_tabfit_P{2}'.format(field, galaxy, params[i]),[t,pt])

#bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,\
#    bfbsc, bfrsc, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa= dres.samples[-1]

#np.save(pos_path + '{0}_{1}_tabfit_bfit'.format(field, galaxy),
#        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,
#         bfbsc, bfrsc, bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa, dres.logl[-1]])

bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,\
    bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa= dres.samples[-1]

np.save(pos_path + '{0}_{1}_tabfit_bfit'.format(field, galaxy),
        [bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,
         bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa, dres.logl[-1]])