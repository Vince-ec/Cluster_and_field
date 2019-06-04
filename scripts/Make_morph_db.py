import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_sfh
from glob import glob
import os

### set home for files
hpath = os.environ['HOME'] + '/'

if hpath.split('/')[-2][-1] == 'a':
    goodss_125 = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_125 = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodss_160 = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    goodsn_160 = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    
else:
    goodss_125 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_125 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodss_160 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss_3dhst.v4.1_f125w.60lfit',format='ascii').to_pandas()
    goodsn_160 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn_3dhst.v4.1_f125w.60lfit',format='ascii').to_pandas()

tabfits = pd.read_pickle('../dataframes/fitdb/tabfitdb.pkl')
    
#add n values
n125 = []
n160 = []

n125_f = []
n160_f = []

for i in tabfits.index:
    if tabfits.field[i][1] == 'S':
        n = goodss_125.n[goodss_125.NUMBER == tabfits.id[i]].values        
        n125.append(n)
        n125_f.append(goodss_125.f[goodss_125.NUMBER == tabfits.id[i]].values)
        
        n = goodss_160.n[goodss_160.NUMBER == tabfits.id[i]].values        
        n160.append(n)
        n160_f.append(goodss_160.f[goodss_160.NUMBER == tabfits.id[i]].values)
        
    if tabfits.field[i][1] == 'N':
        n = goodsn_125.n[goodsn_125.NUMBER == tabfits.id[i]].values
        n125.append(n)
        n125_f.append(goodsn_125.f[goodsn_125.NUMBER == tabfits.id[i]].values)

        n = goodsn_160.n[goodsn_160.NUMBER == tabfits.id[i]].values
        n160.append(n)
        n160_f.append(goodsn_160.f[goodsn_160.NUMBER == tabfits.id[i]].values)

tabfits['n_f125'] = np.array(n125)
tabfits['n_f160'] = np.array(n160)

tabfits['n_f125_f'] = np.array(n125_f)
tabfits['n_f160_f'] = np.array(n160_f)

#add Reff values
Reff125 = []
Reff160 = []

for i in tabfits.index:

    if tabfits.field[i][1] == 'S':
        r = goodss_125.re[goodss_125.NUMBER == tabfits.id[i]].values * np.sqrt(goodss_125.q[goodss_125.NUMBER == tabfits.id[i]].values)
        Reff125.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)
         
        r = goodss_160.re[goodss_160.NUMBER == tabfits.id[i]].values * np.sqrt(goodss_160.q[goodss_160.NUMBER == tabfits.id[i]].values)
        Reff160.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)
            
    if tabfits.field[i][1] == 'N':
        r = goodsn_125.re[goodsn_125.NUMBER == tabfits.id[i]].values * np.sqrt(goodsn_125.q[goodsn_125.NUMBER == tabfits.id[i]].values)
        Reff125.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)

        r = goodsn_160.re[goodsn_160.NUMBER == tabfits.id[i]].values * np.sqrt(goodsn_160.q[goodsn_160.NUMBER == tabfits.id[i]].values)
        Reff160.append(r[0] / cosmo.arcsec_per_kpc_proper(tabfits.zgrism[i]).value)
        
tabfits['Re_f125'] = np.array(Reff125)
tabfits['Re_f160'] = np.array(Reff160)

#add compactness
def A_value(Reff, mass):
    return (Reff) / (mass / 1E11)**0.75

def B_value(Reff, mass):
    return np.log10(mass / Reff**1.5)

c_A125 = np.repeat('n', len(tabfits))
c_A160 = np.repeat('n', len(tabfits))

c_B125 = np.repeat('n', len(tabfits))
c_B160 = np.repeat('n', len(tabfits))

Aval125 = np.repeat(-99.0, len(tabfits))
Aval160 = np.repeat(-99.0, len(tabfits))

Bval125 = np.repeat(-99.0, len(tabfits))
Bval160 = np.repeat(-99.0, len(tabfits))

for i in tabfits.index:
    A = A_value(tabfits.Re_f125[i], 10**tabfits.lmass[i])
    B = B_value(tabfits.Re_f125[i], 10**tabfits.lmass[i])
    
    Aval125[i] = (A)
    Bval125[i] = (B)
        
    if A < 2.5:
        c_A125[i] = 'c'
    else:
        c_A125[i] = 'e'
        
    if B > 10.3:
        c_B125[i] = 'c'
    else:
        c_B125[i] = 'e'
        
    A = A_value(tabfits.Re_f160[i], 10**tabfits.lmass[i])
    B = B_value(tabfits.Re_f160[i], 10**tabfits.lmass[i])
    
    Aval160[i] = (A)
    Bval160[i] = (B)
        
    if A < 2.5:
        c_A160[i] = 'c'
    else:
        c_A160[i] = 'e'
        
    if B > 10.3:
        c_B160[i] = 'c'
    else:
        c_B160[i] = 'e'
        
        
tabfits['compact_A_f125'] = c_A125
tabfits['A_f125'] = Aval125
tabfits['compact_B_f125'] = c_B125
tabfits['B_f125'] = Bval125

tabfits['compact_A_f160'] = c_A160
tabfits['A_f160'] = Aval160
tabfits['compact_B_f160'] = c_B160
tabfits['B_f160'] = Bval160


#add sigma1

def Fphot(field, galaxy_id, phot):
    if phot.lower() == 'f125':
        bfilters = 203
    if phot.lower() == 'f160':
        bfilters = 205

    W, F, E, FLT = np.load('../phot/{0}_{1}_phot.npy'.format(field, galaxy_id))

    return (F[FLT == bfilters] * W[FLT == bfilters]**2 / 3E18)[0]

def IR_prime(n, Reff, R):
    b = 2*n - (1/3)  
    return R * np.exp(-b * (R / Reff)**(1/n))

def Sigma_1(field, galaxy, filt, gfit_cat):
    grow = tabfits.query('id == {0}'.format(galaxy))
    Reff = grow['Re_{0}'.format(filt)].values[0]
    n = grow['n_{0}'.format(filt)].values[0]
    mass = 10**grow['lmass'].values[0]
    
    mgal = gfit_cat.query('NUMBER == {0}'.format(galaxy)).mag.values[0]    
    
    r_range = np.linspace(0,1,1000)
    top = np.trapz(IR_prime(n, Reff, r_range), r_range)
    
    r_range = np.linspace(0,100,100000)
    bottom = np.trapz(IR_prime(n, Reff, r_range), r_range)   
    
    Lgal = 10**((mgal + 48.6) / -2.5)   
    return (top / bottom)*(Lgal / Fphot(field, galaxy, filt))*mass / np.pi**2

S1f125 = []
S1f160 = []

for i in tabfits.index:

    if tabfits.field[i][1] == 'S':
        S1f125.append(Sigma_1('GSD', tabfits.id[i], 'f125', goodss_125))
        S1f160.append(Sigma_1('GSD', tabfits.id[i], 'f160', goodss_160))
    
    if tabfits.field[i][1] == 'N':
        S1f125.append(Sigma_1('GND', tabfits.id[i], 'f125', goodsn_125))
        S1f160.append(Sigma_1('GND', tabfits.id[i], 'f160', goodsn_160))

tabfits['Sigma1_f125'] = np.array(S1f125)
tabfits['Sigma1_f160'] = np.array(S1f160)

c_S1f125 = np.repeat('n', len(tabfits))
c_S1f160 = np.repeat('n', len(tabfits))

for i in tabfits.index:
    A = np.log10(tabfits['Sigma1_f125'][i])
    B = np.log10(tabfits['Sigma1_f160'][i])
    
    if A >= 9.5:
        c_S1f125[i] = 'c'
    else:
        c_S1f125[i] = 'e'
        
    if B >= 9.5:
        c_S1f160[i] = 'c'
    else:
        c_S1f160[i] = 'e'

tabfits['compact_Sigma1_f125'] = np.array(c_S1f125)
tabfits['compact_Sigma1_f160'] = np.array(c_S1f160)

Re = []
A = []
B = []
S1 = []
c_a = []
c_b = []
c_S1 = []
n = []
n_f = [] 
for i in tabfits.index:
    if tabfits.zgrism[i] <= 1.5:
        Re.append(tabfits.Re_f125[i])
        A.append(tabfits.A_f125[i])
        B.append(tabfits.B_f125[i])
        S1.append(tabfits.Sigma1_f125[i])
        c_a.append(tabfits.compact_A_f125[i])
        c_b.append(tabfits.compact_B_f125[i])
        c_S1.append(tabfits.compact_Sigma1_f125[i])
        n.append(tabfits.n_f125[i])
        n_f.append(tabfits.n_f125_f[i])
        
    else:
        Re.append(tabfits.Re_f160[i])
        A.append(tabfits.A_f160[i])
        B.append(tabfits.B_f160[i])
        S1.append(tabfits.Sigma1_f160[i])
        c_a.append(tabfits.compact_A_f160[i])
        c_b.append(tabfits.compact_B_f160[i])
        c_S1.append(tabfits.compact_Sigma1_f160[i])
        n.append(tabfits.n_f160[i])
        n_f.append(tabfits.n_f160_f[i])
        
tabfits['Re'] = np.array(Re)
tabfits['A'] = np.array(A)
tabfits['B'] = np.array(B)
tabfits['Sigma1'] = np.array(S1)
tabfits['compact_A'] = np.array(c_a)
tabfits['compact_B'] = np.array(c_b)
tabfits['compact_Sigma1'] = np.array(c_S1)
tabfits['n'] = np.array(n)
tabfits['n_f'] = np.array(n_f)

tabfits.to_pickle('../dataframes/fitdb/tabfitdb.pkl')