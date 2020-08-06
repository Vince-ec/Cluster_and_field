#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
import pandas as pd
import dynesty 
import os
import sys
import numpy as np
from spec_id import Gaussian_prior
from multiprocessing import Pool
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    lsig = float(sys.argv[1]) 
    hsig = float(sys.argv[2])
    z50 = float(sys.argv[3])
    zlow = float(sys.argv[4])
    zhi = float(sys.argv[5])
    
def sym_err(DB, param):
    s_err = []
    for i in DB.index:
        hdr = DB['{}_hdr'.format(param)][i]
        s_err.append((hdr[1] - hdr[0])/2)    
    return np.array(s_err)

Hdb = pd.read_pickle(data_path + 'hubble_db.pkl') 
Cdb = Hdb.query(' {} < log_Sigma1 < {} and sf_prob_2 < 0.2 and z_50 > {} and {} < zgrism < {}'.format(lsig, hsig, z50, zlow, zhi)).sort_values('zgrism') 

rshifts = Cdb.zgrism.values
t50s = Cdb.t_50.values
serr = sym_err(Cdb, 't_50')
    
def ln_likelihood(pars):
    """ The likelihood function evaluation requires a particular set of model parameters and the data """
    H, toff, Va, lnVb = pars
    
    cosmo = FlatLambdaCDM(H0=H, Om0=0.3)
    V = -Va * rshifts + np.exp(lnVb)
    V[V<0] = 0

    N = len(t50s)
    dy = t50s - (cosmo.age(rshifts).value - toff)
    ivar = 1 / (serr**2 + V) # inverse-variance now includes intrinsic scatter
    
    return -0.5 * (N*np.log(2*np.pi) - np.sum(np.log(ivar)) + np.sum(dy**2 * ivar))

def prior(u):
    H = Gaussian_prior(u[0], [50, 90], 70, 5)
    toff = 3*u[1]
    Va = Gaussian_prior(u[2], [-0.5, 0.5], 0, 0.25)
    lnVb = Gaussian_prior(u[3], [0, -3], -1.5, 0.5)
    
    return H, toff, Va, lnVb

sampler = dynesty.DynamicNestedSampler(ln_likelihood, prior, ndim = 4, 
                    bound = 'multi',pool=Pool(processes=8), queue_size=8)
sampler.run_nested(wt_kwargs={'pfrac': 1.0}, dlogz_init=0.01, nlive_init=1000, print_progress=True)

dres = sampler.results
np.save(out_path + '{}_lS1_{}_z50_{}_{}_z_{}'.format(lsig, hsig, z50, zlow, zhi), dres) 