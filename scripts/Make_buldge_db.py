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
    goodss_rad = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_rad = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
else:
    goodss_rad = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_rad = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()

###### create tabit db#########
GS_SF = pd.read_pickle('../dataframes/galaxy_frames/GS_buldge_SFup.pkl')
GS_Q = pd.read_pickle('../dataframes/galaxy_frames/GS_buldge_Qup.pkl')
GN_SF = pd.read_pickle('../dataframes/galaxy_frames/GN_buldge_SFup.pkl')
GN_Q = pd.read_pickle('../dataframes/galaxy_frames/GN_buldge_Qup.pkl')

GS_cat = pd.concat([GS_SF,GS_Q])
GS_cat['field'] = np.repeat('GSD', len(GS_cat))

GN_cat = pd.concat([GN_SF,GN_Q])
GN_cat['field'] = np.repeat('GND', len(GN_cat))

select = pd.concat([GS_cat, GN_cat])
#####parameters to get Z, m1-10, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {}
params = ['m', 'lm', 'z', 'd', 'lwa']
k = ['Z', 'lmass', 'zgrism', 'Av', 'lwa']

for p in range(len(params)):
    m = np.repeat(-99.0,len(select))
    hci = []
    offmass = []
    
    for i in range(len(select.index)):
        try:
            flist = glob('../data/posteriors/{}_{}_*_P{}.npy'.format(select.field[select.index[i]], select.id[select.index[i]],params[p]))

            for f in flist:
                ext = os.path.basename(f).strip('{}_{}_'.format(select.field[select.index[i]], select.id[select.index[i]])).strip('_P{}.npy'.format(params[p]))
                if ext in ['tabfit', 'SFfit_p1']:
                    print(f)

            x,px = np.load(f)            
            m[i], mreg, oreg = Highest_density_region(px,x)
            hci.append(mreg)
            offmass.append(oreg)
        except:
            hci.append([0])
            offmass.append([0])
    
    fitvals['{0}'.format(k[p])] = m
    fitvals['{0}_hci'.format(k[p])] = hci
    fitvals['{0}_modality'.format(k[p])] = offmass
    
#make db
tabfits = pd.DataFrame(fitvals)
tabfits['field'] = select.field.values
tabfits['id'] = select.id.values
tabfits['z_grizli'] = select.z_50.values

#add SFH values
z_50= np.repeat(-99.0,len(tabfits))
z_50_hci = []
z_50_oreg = []
z_80= np.repeat(-99.0,len(tabfits))
z_80_hci = []
z_80_oreg = []
z_90= np.repeat(-99.0,len(tabfits))
z_90_hci = []
z_90_oreg = []
t_50= np.repeat(-99.0,len(tabfits))
t_50_hci = []
t_50_oreg = []
t_80= np.repeat(-99.0,len(tabfits))
t_80_hci = []
t_80_oreg = []
t_90= np.repeat(-99.0,len(tabfits))
t_90_hci = []
t_90_oreg = []
log_ssfr= np.repeat(-99.0,len(tabfits))
log_ssfr_hci = []
log_ssfr_oreg = []
for i in range(len(tabfits.index)):
    try:
        sfh = Gen_sfh(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]], tabfits.z_grizli[tabfits.index[i]] ,5000)
        np.save('../data/SFH/{}_{}.npy'.format(tabfits.field[tabfits.index[i]], tabfits.id[tabfits.index[i]]), 
                [sfh.LBT, sfh.SFH])
        print(sfh.z_50)
        z_50[i] = sfh.z_50
        z_50_hci.append(sfh.z_50_hci)
        z_50_oreg.append(sfh.z_50_offreg)

        z_80[i] = sfh.z_80
        z_80_hci.append(sfh.z_80_hci)
        z_80_oreg.append(sfh.z_80_offreg)

        z_90[i] = sfh.z_90
        z_90_hci.append(sfh.z_90_hci)
        z_90_oreg.append(sfh.z_90_offreg)

        t_50[i] = sfh.t_50
        t_50_hci.append(sfh.t_50_hci)
        t_50_oreg.append(sfh.t_50_offreg)

        t_80[i] = sfh.t_80
        t_80_hci.append(sfh.t_80_hci)
        t_80_oreg.append(sfh.t_80_offreg)

        t_90[i] = sfh.t_90
        t_90_hci.append(sfh.t_90_hci)
        t_90_oreg.append(sfh.t_90_offreg)

        log_ssfr[i] = sfh.lssfr
        log_ssfr_hci.append(sfh.lssfr_hci)
        log_ssfr_oreg.append(sfh.lssfr_offreg)
        
    except:
        z_50_hci.append(np.array([0]))
        z_80_hci.append(np.array([0]))
        z_90_hci.append(np.array([0]))
        t_50_hci.append([0])
        t_80_hci.append([0])
        t_90_hci.append([0])
        log_ssfr_hci.append([0])
        
        z_50_oreg.append(np.array([0]))
        z_80_oreg.append(np.array([0]))
        z_90_oreg.append(np.array([0]))
        t_50_oreg.append([0])
        t_80_oreg.append([0])
        t_90_oreg.append([0])
        log_ssfr_oreg.append([0])
        
tabfits['z_50'] = z_50
tabfits['z_50_hci'] = z_50_hci
tabfits['z_50_modality'] = z_50_oreg

tabfits['z_80'] = z_80
tabfits['z_80_hci'] = z_80_hci
tabfits['z_80_modality'] = z_80_oreg

tabfits['z_90'] = z_90
tabfits['z_90_hci'] = z_90_hci
tabfits['z_90_modality'] = z_90_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_80'] = t_80
tabfits['t_80_hci'] = t_80_hci
tabfits['t_80_modality'] = t_80_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hci'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_90'] = t_90
tabfits['t_90_hci'] = t_90_hci
tabfits['t_90_modality'] = t_90_oreg

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hci'] = log_ssfr_hci
tabfits['log_ssfr_modality'] = log_ssfr_oreg

tabfits.to_pickle('../dataframes/fitdb/buldgefitsdb.pkl')

if hpath.split('/')[-2][-1] == 'a':
    goodss_125 = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_125 = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodss_160 = Table.read('/Users/vestrada/Downloads/allfields/goodss/goodss_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    goodsn_160 = Table.read('/Users/vestrada/Downloads/allfields/goodsn/goodsn_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    
else:
    goodss_125 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss/goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_125 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn/goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodss_160 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodss/goodss_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    goodsn_160 = Table.read('/Users/Vince.ec/Clear_data/galaxy_meas/goodsn/goodsn_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()

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

Reff125_sig = []
Reff160_sig = []

for i in tabfits.index:
    if tabfits.zgrism[i] < 0:
        redshift = tabfits.z_grizli[i]
    else:
        redshift = tabfits.zgrism[i]

    
    
    if tabfits.field[i][1] == 'S':   
        r = goodss_125.re[goodss_125.NUMBER == tabfits.id[i]].values
        q = goodss_125.q[goodss_125.NUMBER == tabfits.id[i]].values
        rs = goodss_125.dre[goodss_125.NUMBER == tabfits.id[i]].values
        qs = goodss_125.dq[goodss_125.NUMBER == tabfits.id[i]].values
        
        Reff125.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(redshift).value)
        Reff125_sig.append(np.sqrt(q*qs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(redshift).value)
        
        r = goodss_160.re[goodss_160.NUMBER == tabfits.id[i]].values
        q = goodss_160.q[goodss_160.NUMBER == tabfits.id[i]].values
        rs = goodss_160.dre[goodss_160.NUMBER == tabfits.id[i]].values
        qs = goodss_160.dq[goodss_160.NUMBER == tabfits.id[i]].values
        
        Reff160.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(redshift).value)
        Reff160_sig.append(np.sqrt(q*qs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(redshift).value)
            
    if tabfits.field[i][1] == 'N':
        r = goodsn_125.re[goodsn_125.NUMBER == tabfits.id[i]].values
        q = goodsn_125.q[goodsn_125.NUMBER == tabfits.id[i]].values
        rs = goodsn_125.dre[goodsn_125.NUMBER == tabfits.id[i]].values
        qs = goodsn_125.dq[goodsn_125.NUMBER == tabfits.id[i]].values
        
        Reff125.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(redshift).value)
        Reff125_sig.append(np.sqrt(q*qs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(redshift).value)
        
        r = goodsn_160.re[goodsn_160.NUMBER == tabfits.id[i]].values
        q = goodsn_160.q[goodsn_160.NUMBER == tabfits.id[i]].values
        rs = goodsn_160.dre[goodsn_160.NUMBER == tabfits.id[i]].values
        qs = goodsn_160.dq[goodsn_160.NUMBER == tabfits.id[i]].values
        
        Reff160.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(redshift).value)
        Reff160_sig.append(np.sqrt(q*qs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(redshift).value)
        
        
tabfits['Re_f125'] = np.array(Reff125)
tabfits['Re_f160'] = np.array(Reff160)
tabfits['Re_f125_sig'] = np.array(Reff125_sig)
tabfits['Re_f160_sig'] = np.array(Reff160_sig)

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
    return (top / bottom)*(Lgal / Fphot(field, galaxy, filt))*mass / np.pi

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
Re_sig = []
A = []
B = []
S1 = []
c_a = []
c_b = []
c_S1 = []
n = []
n_f = [] 
for i in tabfits.index:
    if tabfits.zgrism[i] < 0:
        redshift = tabfits.z_grizli[i]
    else:
        redshift = tabfits.zgrism[i]
    
    if redshift <= 1.5:
        Re.append(tabfits.Re_f125[i])
        Re_sig.append(tabfits.Re_f125_sig[i])
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
        Re_sig.append(tabfits.Re_f160_sig[i])
        A.append(tabfits.A_f160[i])
        B.append(tabfits.B_f160[i])
        S1.append(tabfits.Sigma1_f160[i])
        c_a.append(tabfits.compact_A_f160[i])
        c_b.append(tabfits.compact_B_f160[i])
        c_S1.append(tabfits.compact_Sigma1_f160[i])
        n.append(tabfits.n_f160[i])
        n_f.append(tabfits.n_f160_f[i])
        
tabfits['Re'] = np.array(Re)
tabfits['Re_sig'] = np.array(Re_sig)
tabfits['A'] = np.array(A)
tabfits['B'] = np.array(B)
tabfits['Sigma1'] = np.array(S1)
tabfits['compact_A'] = np.array(c_a)
tabfits['compact_B'] = np.array(c_b)
tabfits['compact_Sigma1'] = np.array(c_S1)
tabfits['n'] = np.array(n)
tabfits['n_f'] = np.array(n_f)

tabfits.to_pickle('../dataframes/fitdb/buldgefitsdb.pkl')