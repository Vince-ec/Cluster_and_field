import numpy as np
import pandas as pd 
import os
from astropy.table import Table
from astropy.io import fits
from glob import glob
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    phot_path = '../phot/'


def Phi(r, re, n):
    return r*np.exp(-(2*n - 1/3) * ((r/re)**(1/n) - 1))

def d_Phi_re(r, re, n):
    P = (2 - 1/(3*n)) * (1/re)**((1/n) + 1) * r**(1/n)
    return P*Phi(r, re, n)

def d_Phi_n(r, re, n):
    P = (2 - 2 * (r/re)**(1/n)) + ((2*n - 1/3) / n**2) * (r/re)**(1/n) * np.log(r/re)
    return P*Phi(r, re, n)

def g(re, n):
    r1 = np.arange(0.001,1,0.001)    
    return np.trapz(Phi(r1, re, n), r1)

def h(re, n):
    r = np.arange(0.001,1000,0.001)    
    return np.trapz(Phi(r, re, n), r)

def Theta(re, n):
    return g(re, n) / h(re, n)

def Pi(Lg, Lp, M):
    return (Lg/Lp) * (M/np.pi)

def Sigma1(re, n, Lg, Lp, M):
    return Theta(re, n) * Pi(Lg, Lp, M)

def g_prime_re(re, n):
    r1 = np.arange(0.001,1,0.001)    
    return np.trapz(d_Phi_re(r1, re, n), r1)

def h_prime_re(re, n):
    r = np.arange(0.001,1000,0.001)    
    return np.trapz(d_Phi_re(r, re, n), r)

def g_prime_n(re, n):
    r1 = np.arange(0.001,1,0.001)    
    return np.trapz(d_Phi_n(r1, re, n), r1)

def h_prime_n(re, n):
    r = np.arange(0.001,1000,0.001)    
    return np.trapz(d_Phi_n(r, re, n), r)

def d_Theta_re(re, n):
    return (g_prime_re(re, n) * h(re, n) - g(re, n) * h_prime_re(re, n)) / h(re, n)**2

def d_Theta_n(re, n):
    return (g_prime_n(re, n) * h(re, n) - g(re, n) * h_prime_n(re, n)) / h(re, n)**2

def d_Pi_Lg(Lg, Lp, M):
    return (1/Lp) * (M/np.pi)

def d_Pi_Lp(Lg, Lp, M):
    return -(Lg/Lp**2) * (M/np.pi)

def d_Pi_M(Lg, Lp, M):
    return (Lg/Lp) * (1/np.pi)

def re_term(re, n, Lg, Lp, M, sig_re):
    return d_Theta_re(re, n) * Pi(Lg, Lp, M) * sig_re

def n_term(re, n, Lg, Lp, M, sig_n):
    return d_Theta_n(re, n) * Pi(Lg, Lp, M) * sig_n

def Lg_term(re, n, Lg, Lp, M, sig_Lg):
    return d_Pi_Lg(Lg, Lp, M) * Theta(re, n) * sig_Lg

def Lp_term(re, n, Lg, Lp, M, sig_Lp):
    return d_Pi_Lp(Lg, Lp, M) * Theta(re, n) * sig_Lp

def M_term(re, n, Lg, Lp, M, sig_M):
    return d_Pi_M(Lg, Lp, M) * Theta(re, n) * sig_M

def Sigma1_sig(re, n, Lg, Lp, M, sig_re, sig_n, sig_Lg, sig_Lp, sig_M):
    return np.sqrt(re_term(re, n, Lg, Lp, M, sig_re)**2 + n_term(re, n, Lg, Lp, M, sig_n)**2 +\
        Lg_term(re, n, Lg, Lp, M, sig_Lg)**2 + Lp_term(re, n, Lg, Lp, M, sig_Lp)**2 + M_term(re, n, Lg, Lp, M, sig_M)**2)

####################################

def Fphot(field, galaxy_id, phot):
    if phot.lower() == 'f125':
        bfilters = 203
    if phot.lower() == 'f160':
        bfilters = 205

    W, F, E, FLT = np.load(phot_path + '{0}_{1}_phot.npy'.format(field, galaxy_id))

    return (F[FLT == bfilters] * W[FLT == bfilters]**2 / 3E18)[0], (E[FLT == bfilters] * W[FLT == bfilters]**2 / 3E18)[0]

def Extract_params(field, gid, redshift, DF):
    if redshift < 1.5:
        Filt = 'f125'
    else:
        Filt = 'f160'
    
    re, n, mag, lM, dre, dn, dmag, dlM = DF.query('id == {}'.format(gid))[['Re','n','mag', 'lmass', 'Re_sig', 'n_sig', 'mag_sig', 'lmass_hdr']].values[0]
    
    M = 10**lM
    dM = M * np.log(10) * (dlM[1] - dlM[0])/2

    Lg = 10**((mag + 48.6) / -2.5)   
    dLg = Lg * dmag / mag 

    Lp, dLp = Fphot(field, gid, Filt)
    return re, n, Lg, Lp, M, dre, dn, dLg, dLp, dM

