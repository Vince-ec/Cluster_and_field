#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
import pandas as pd
import dynesty 
import os
import sys
import numpy as np
from spec_id import Gaussian_prior
from multiprocessing import Pool
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    num = int(sys.argv[1]) 

out_path = '/scratch/user/vestrada78840/chidat/'

ndx = [0.6, 0.61796432, 0.71674852, 0.81553271, 0.91431691, 1.01310111, 1.1118853,
 1.2106695,  1.30945369, 1.40823789, 1.50702209, 1.60580628, 1.70459048,
 1.80337468, 1.90215887, 2.00094307, 2.09972727, 2.19851146, 2.29729566,
 2.39607986,2.5] 
ndy = [21.73290068,21.73290068, 20.02264362, 18.36194567, 16.76407205, 15.2553624,  14.05997006,
 13.43233581, 12.692396,   11.95048395, 10.98004395, 10.13452753,  9.36988171,
  9.04101404,  9.24397231,  9.46474917,  9.71777641, 10.01323645, 10.37144486,
 10.81226734, 10.81226734]

isnr = interp1d(ndx, ndy)

def sim_pop(num_sim_gals, H0, toff, V, serr, zrange):
    z = np.linspace(0.3,3,1000)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    limcurve = cosmo.age(z).value

    cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
    curve = cosmo.age(z).value - toff

    zdist = np.random.rand(num_sim_gals)*(zrange[1] - zrange[0]) + zrange[0]
    icurve = interp1d(z, curve)(zdist)
    ilimcurve = interp1d(z, limcurve)(zdist)

    Voffset = []

    while len(Voffset) < num_sim_gals:
        initoff = np.random.normal(loc = 0, scale= np.sqrt(V))
        if 0 < icurve[len(Voffset)] + initoff < ilimcurve[len(Voffset)]:
            Voffset.append(initoff)
        else:
            pass
      
    initsnr = []
    print('at snr')
    while len(initsnr) < num_sim_gals:
        initerr = np.random.normal(isnr(zdist[len(initsnr)]), 5.5)
        if initerr > 5:
            initsnr.append(initerr)
        else:
            pass
    rerr = (icurve + Voffset)/np.array(initsnr)
    Soffset = []

    while len(Soffset) < num_sim_gals:
        initoff = np.random.normal(loc = 0, scale= rerr[len(Soffset)])
        if 0 < icurve[len(Soffset)] + Voffset[len(Soffset)] + initoff < ilimcurve[len(Soffset)]:
            Soffset.append(initoff)
        else:
            pass
    
    return zdist, icurve + np.array(Voffset) + np.array(Soffset), rerr
    
sim_z, sim_t50, sim_err = sim_pop(num, 74.62, 1.22, 0.44**2, [0.5,2.5])

def sim_ln_likelihood(pars):
    """ The likelihood function evaluation requires a particular set of model parameters and the data """
    H, toff, Va, lnVb = pars
    
    cosmo = FlatLambdaCDM(H0=H, Om0=0.3)
    V = -Va * sim_z + np.exp(lnVb)
    V[V<0] = 0

    N = len(sim_t50)
    dy = sim_t50 - (cosmo.age(sim_z).value - toff)
    ivar = 1 / (sim_err**2 + V) # inverse-variance now includes intrinsic scatter
    
    return -0.5 * (N*np.log(2*np.pi) - np.sum(np.log(ivar)) + np.sum(dy**2 * ivar))

def sim_prior(u):
    H = Gaussian_prior(u[0], [50, 90], 70, 5)
    toff = 3*u[1]
    Va = Gaussian_prior(u[2], [-0.5, 0.5], 0, 0.25)
    lnVb = Gaussian_prior(u[3], [0, -3], -1.5, 0.5)
    
    return H, toff, Va, lnVb

sampler = dynesty.DynamicNestedSampler(sim_ln_likelihood, sim_prior, ndim = 4, 
                    bound = 'multi',pool=Pool(processes=8), queue_size=8)
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, nlive_init=1000, print_progress=False)

dres = sampler.results
np.save(out_path + 'sim_H0_n{}'.format(num), dres) 
