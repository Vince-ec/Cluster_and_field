import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from spec_tools import Rescale_sfh
from glob import glob
import os
import pickle
from make_sfh_tool import Gen_sim_SFH

### set home for files
hpath = os.environ['HOME'] + '/'

#TODO: upload van der wel catalogs to cluster 

if hpath == '/home/vestrada78840/':
    data_path = '/scratch/user/vestrada78840/data/'
    pos_path = '/home/vestrada78840/posteriors/'
    sfh_path = '/scratch/user/vestrada78840/SFH/'
    phot_path = '/scratch/user/vestrada78840/phot/'
    db_path = '/scratch/user/vestrada78840/data/'
    ### set home for files
    goodss_125 = Table.read(data_path + 'goodss_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodsn_125 = Table.read(data_path + 'goodsn_3dhst.v4.1_f125w.galfit',format='ascii').to_pandas()
    goodss_160 = Table.read(data_path + 'goodss_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    goodsn_160 = Table.read(data_path + 'goodsn_3dhst.v4.1_f160w.galfit',format='ascii').to_pandas()
    
else:
    data_path = '../data/'
    pos_path = '../data/posteriors/'
    sfh_path = '../data/SFH/'
    phot_path = '../phot/'
    db_path = '../dataframes/galaxy_frames/'

#####parameters to get Z, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {'field':[], 'id':[],'Z':[], 'Z_hdr':[],'Z_modality':[],'lmass':[], 'lmass_hdr':[],
           'lmass_modality':[], 'Av':[], 'Av_hdr':[],'Av_modality':[],
           'zgrism':[], 'zgrism_hdr':[],'zgrism_modality':[],
           'lwa':[], 'lwa_hdr':[],'lwa_modality':[],
           'lwa_u':[], 'lwa_u_hdr':[],'lwa_u_modality':[],
           'lwa_r':[], 'lwa_r_hdr':[],'lwa_r_modality':[],
           'bfm':[], 'bfa':[], 'bfm1':[], 
           'bfm2':[], 'bfm3':[], 'bfm4':[], 'bfm5':[], 'bfm6':[], 
           'bfm7':[], 'bfm8':[], 'bfm9':[], 'bfm10':[], 
            'bflm':[],'bfz':[], 'bfd':[], 'bfbp1':[], 'bfrp1':[], 'bfba':[], 
           'bfbb':[], 'bfbl':[], 'bfra':[], 'bfrb':[], 'bfrl':[]}

k = ['Z','lmass', 'Av', 'zgrism', 'lwa', 'lwa_u', 'lwa_r','bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
     'bfm7', 'bfm8', 'bfm9', 'bfm10',       
     'bflm','bfz', 'bfd', 'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
params = ['m','lm', 'd','z', 'lwa', 'lwa_u', 'lwa_r','bfm', 'bfa', 'bfm1', 'bfm2', 'bfm3', 'bfm4', 'bfm5', 'bfm6', 
          'bfm7', 'bfm8', 'bfm9', 'bfm10',  
            'bflm', 'bfz', 'bfd', 'bfbp1', 'bfrp1', 'bfba', 'bfbb', 'bfbl', 'bfra', 'bfrb', 'bfrl']
P_params = ['Pm','Plm', 'Pd', 'Pz', 'Plwa', 'Plwa_u', 'Plwa_r','-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 
            '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']


NGSID = np.load(data_path + 'N_GSD_2.npy',  allow_pickle=True)
NGNID = np.load(data_path + 'N_GND_2.npy',  allow_pickle=True)
NGSz = np.load(data_path + 'N_GSD_2_z.npy', allow_pickle=True)
NGNz = np.load(data_path + 'N_GND_2_z.npy', allow_pickle=True)

NGSsf = np.load(data_path + 'N_GSD_2_sf.npy',  allow_pickle=True)
NGNsf = np.load(data_path + 'N_GND_2_sf.npy',  allow_pickle=True)


###### create tabit db#########
for i in range(len(NGSID)):
    fitvals['field'].append('GSD')
    fitvals['id'].append(NGSID[i])
    
    if NGSsf[i] == 'Q':
        fit_db = np.load(pos_path + 'GSD_{}_tabfit.npy'.format(NGSID[i]),allow_pickle=True).item()
    else:
        fit_db = np.load(pos_path + 'GSD_{}_SFfit_p1_fits.npy'.format(NGSID[i]),allow_pickle=True).item()

    for p in range(len(params)):
        if NGSsf[i] == 'Q':
            if P_params[p] == '-': 
                fitvals[k[p]].append(fit_db[params[p]])
            else:
                m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                fitvals['{}'.format(k[p])].append(m)
                fitvals['{}_hdr'.format(k[p])].append(mreg)
                fitvals['{}_modality'.format(k[p])].append(oreg)

        else:
            if P_params[p] == '-':
                if params[p] in ['bfz', 'bfm7', 'bfm8', 'bfm9', 'bfm10']:
                    fitvals[k[p]].append(-99)
                else:
                    fitvals[k[p]].append(fit_db[params[p]])
            else:
                if params[p] == 'z':
                    fitvals['{}'.format(k[p])].append(NGSz[i])
                    fitvals['{}_hdr'.format(k[p])].append(-99)
                    fitvals['{}_modality'.format(k[p])].append(-99)      

                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)

for i in range(len(NGNID)):
    fitvals['field'].append('GND')
    fitvals['id'].append(NGNID[i])
    if NGNsf[i] == 'Q':
        fit_db = np.load(pos_path + 'GND_{}_tabfit.npy'.format(NGNID[i]),allow_pickle=True).item()
    else:
        fit_db = np.load(pos_path + 'GND_{}_SFfit_p1_fits.npy'.format(NGNID[i]),allow_pickle=True).item()
        
        
    for p in range(len(params)):
        if NGNsf[i] == 'Q':
            if P_params[p] == '-':
                fitvals[k[p]].append(fit_db[params[p]])
            else:
                m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                fitvals['{}'.format(k[p])].append(m)
                fitvals['{}_hdr'.format(k[p])].append(mreg)
                fitvals['{}_modality'.format(k[p])].append(oreg)

        else:
            if P_params[p] == '-':
                if params[p] in ['bfz', 'bfm7', 'bfm8', 'bfm9', 'bfm10']:
                    fitvals[k[p]].append(-99)
                else:
                    fitvals[k[p]].append(fit_db[params[p]])
            else:
                if params[p] == 'z':
                    fitvals['{}'.format(k[p])].append(NGNz[i])
                    fitvals['{}_hdr'.format(k[p])].append(-99)
                    fitvals['{}_modality'.format(k[p])].append(-99)      

                else:
                    m, mreg, oreg = Highest_density_region(fit_db[P_params[p]],fit_db[params[p]])
                    fitvals['{}'.format(k[p])].append(m)
                    fitvals['{}_hdr'.format(k[p])].append(mreg)
                    fitvals['{}_modality'.format(k[p])].append(oreg)
                
#make db
for k in fitvals.keys():
    print(k, len(fitvals[k]))

tabfits = pd.DataFrame(fitvals)

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

for i in range(len(tabfits)):
    if tabfits.bfm10[i] > -1:
        print(tabfits.field[i], tabfits.id[i],i)
        with open(glob(sfh_path + '{}_{}_1D.pkl'.format(tabfits.field[i], tabfits.id[i]))[0], 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)    
    else:
        print(tabfits.field[i], tabfits.id[i],i)
        with open(glob(sfh_path + '{}_{}_p1_1D.pkl'.format(tabfits.field[i], int(tabfits.id[i])))[0], 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

    np.save(sfh_path + '{}_{}'.format(tabfits.field[i], tabfits.id[i]),[sfh.LBT, sfh.SFH],allow_pickle=True)
    np.save(sfh_path + '{}_{}_16'.format(tabfits.field[i], tabfits.id[i]),[sfh.LBT, sfh.SFH_16],allow_pickle=True)
    np.save(sfh_path + '{}_{}_84'.format(tabfits.field[i], tabfits.id[i]),[sfh.LBT, sfh.SFH_84],allow_pickle=True)

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

    log_ssfr[i] = np.log10((np.trapz(sfh.SFH[:11], sfh.LBT[:11])/0.1) / 10**tabfits.lmass[i])
    log_ssfr_hci.append(sfh.lssfr_hci)
    log_ssfr_oreg.append(sfh.lssfr_offreg)

    
    
tabfits['z_50'] = z_50
tabfits['z_50_hdr'] = z_50_hci
tabfits['z_50_modality'] = z_50_oreg

tabfits['z_80'] = z_80
tabfits['z_80_hdr'] = z_80_hci
tabfits['z_80_modality'] = z_80_oreg

tabfits['z_90'] = z_90
tabfits['z_90_hdr'] = z_90_hci
tabfits['z_90_modality'] = z_90_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hdr'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_80'] = t_80
tabfits['t_80_hdr'] = t_80_hci
tabfits['t_80_modality'] = t_80_oreg

tabfits['t_50'] = t_50
tabfits['t_50_hdr'] = t_50_hci
tabfits['t_50_modality'] = t_50_oreg

tabfits['t_90'] = t_90
tabfits['t_90_hdr'] = t_90_hci
tabfits['t_90_modality'] = t_90_oreg

tabfits['log_ssfr'] = log_ssfr
tabfits['log_ssfr_hdr'] = log_ssfr_hci
tabfits['log_ssfr_modality'] = log_ssfr_oreg
   
#add n values
n125 = []
n160 = []

n125_f = []
n160_f = []

n125_sig = []
n160_sig = []

for i in tabfits.index:
    if tabfits.field[i][1] == 'S':
        n = goodss_125.n[goodss_125.NUMBER == tabfits.id[i]].values  
        nf = goodss_125.f[goodss_125.NUMBER == tabfits.id[i]].values
        ns = goodss_125.dn[goodss_125.NUMBER == tabfits.id[i]].values
        n125.append(n)
        n125_f.append(nf)
        n125_sig.append(ns)
        
        n = goodss_160.n[goodss_160.NUMBER == tabfits.id[i]].values   
        nf = goodss_160.f[goodss_160.NUMBER == tabfits.id[i]].values        
        ns = goodss_160.dn[goodss_160.NUMBER == tabfits.id[i]].values        
        n160.append(n)
        n160_f.append(nf)
        n160_sig.append(ns)
        
    if tabfits.field[i][1] == 'N':
        n = goodsn_125.n[goodsn_125.NUMBER == tabfits.id[i]].values  
        nf = goodsn_125.f[goodsn_125.NUMBER == tabfits.id[i]].values
        ns = goodsn_125.dn[goodsn_125.NUMBER == tabfits.id[i]].values
        n125.append(n)
        n125_f.append(nf)
        n125_sig.append(ns)
        
        n = goodsn_160.n[goodsn_160.NUMBER == tabfits.id[i]].values   
        nf = goodsn_160.f[goodsn_160.NUMBER == tabfits.id[i]].values        
        ns = goodsn_160.dn[goodsn_160.NUMBER == tabfits.id[i]].values        
        n160.append(n)
        n160_f.append(nf)
        n160_sig.append(ns)

tabfits['n_f125'] = np.array(n125)
tabfits['n_f160'] = np.array(n160)

tabfits['n_f125_f'] = np.array(n125_f)
tabfits['n_f160_f'] = np.array(n160_f)

tabfits['n_f125_sig'] = np.array(n125_sig)
tabfits['n_f160_sig'] = np.array(n160_sig)

# add magnitudes
mag125 = []
mag160 = []

mag125_sig = []
mag160_sig = []

for i in tabfits.index:
    if tabfits.field[i][1] == 'S':
        mag = goodss_125.mag[goodss_125.NUMBER == tabfits.id[i]].values  
        mags = goodss_125.dmag[goodss_125.NUMBER == tabfits.id[i]].values
        mag125.append(mag)
        mag125_sig.append(mags)
        
        mag = goodss_160.mag[goodss_160.NUMBER == tabfits.id[i]].values   
        mags = goodss_160.dmag[goodss_160.NUMBER == tabfits.id[i]].values        
        mag160.append(mag)
        mag160_sig.append(mags)
        
    if tabfits.field[i][1] == 'N':
        mag = goodsn_125.mag[goodsn_125.NUMBER == tabfits.id[i]].values  
        mags = goodsn_125.dmag[goodsn_125.NUMBER == tabfits.id[i]].values
        mag125.append(mag)
        mag125_sig.append(mags)
        
        mag = goodsn_160.mag[goodsn_160.NUMBER == tabfits.id[i]].values   
        mags = goodsn_160.dmag[goodsn_160.NUMBER == tabfits.id[i]].values        
        mag160.append(mag)
        mag160_sig.append(mags)

tabfits['mag_f125'] = np.array(mag125)
tabfits['mag_f160'] = np.array(mag160)

tabfits['mag_f125_sig'] = np.array(mag125_sig)
tabfits['mag_f160_sig'] = np.array(mag160_sig)

#add Reff values
Reff125 = []
Reff160 = []
Reff125_sig = []
Reff160_sig = []

Rmaj125 = []
Rmaj160 = []
Rmaj125_sig = []
Rmaj160_sig = []

Rarc125 = []
Rarc160 = []
Rarc125_sig = []
Rarc160_sig = []

for i in tabfits.index:
#    if tabfits.zgrism[i] > 0:
    rshift = tabfits.zgrism[i]
#    else:
#        rshift = tabfits.zfit[i]
    
    if tabfits.field[i][1] == 'S':   
        r = goodss_125.re[goodss_125.NUMBER == tabfits.id[i]].values
        q = goodss_125.q[goodss_125.NUMBER == tabfits.id[i]].values
        rs = goodss_125.dre[goodss_125.NUMBER == tabfits.id[i]].values
        qs = goodss_125.dq[goodss_125.NUMBER == tabfits.id[i]].values
        
        Reff125.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(rshift).value)
        Reff125_sig.append(np.sqrt(q*rs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rmaj125.append(r / cosmo.arcsec_per_kpc_proper(rshift).value)
        Rmaj125_sig.append(rs / cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rarc125.append(r)
        Rarc125_sig.append(rs)
        
        r = goodss_160.re[goodss_160.NUMBER == tabfits.id[i]].values
        q = goodss_160.q[goodss_160.NUMBER == tabfits.id[i]].values
        rs = goodss_160.dre[goodss_160.NUMBER == tabfits.id[i]].values
        qs = goodss_160.dq[goodss_160.NUMBER == tabfits.id[i]].values
        
        Reff160.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(rshift).value)
        Reff160_sig.append(np.sqrt(q*rs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rmaj160.append(r / cosmo.arcsec_per_kpc_proper(rshift).value)
        Rmaj160_sig.append(rs / cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rarc160.append(r)
        Rarc160_sig.append(rs)
        
    if tabfits.field[i][1] == 'N':
        r = goodsn_125.re[goodsn_125.NUMBER == tabfits.id[i]].values
        q = goodsn_125.q[goodsn_125.NUMBER == tabfits.id[i]].values
        rs = goodsn_125.dre[goodsn_125.NUMBER == tabfits.id[i]].values
        qs = goodsn_125.dq[goodsn_125.NUMBER == tabfits.id[i]].values
        
        Reff125.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(rshift).value)
        Reff125_sig.append(np.sqrt(q*rs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rmaj125.append(r / cosmo.arcsec_per_kpc_proper(rshift).value)
        Rmaj125_sig.append(rs / cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rarc125.append(r)
        Rarc125_sig.append(rs)
        
        r = goodsn_160.re[goodsn_160.NUMBER == tabfits.id[i]].values
        q = goodsn_160.q[goodsn_160.NUMBER == tabfits.id[i]].values
        rs = goodsn_160.dre[goodsn_160.NUMBER == tabfits.id[i]].values
        qs = goodsn_160.dq[goodsn_160.NUMBER == tabfits.id[i]].values
        
        Reff160.append((r * np.sqrt(q)) / cosmo.arcsec_per_kpc_proper(rshift).value)
        Reff160_sig.append(np.sqrt(q*rs**2 + r**2/(4*q)*qs**2)/ cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rmaj160.append(r / cosmo.arcsec_per_kpc_proper(rshift).value)
        Rmaj160_sig.append(rs / cosmo.arcsec_per_kpc_proper(rshift).value)
        #
        Rarc160.append(r)
        Rarc160_sig.append(rs)     
        
tabfits['Re_f125'] = np.array(Reff125)
tabfits['Re_f160'] = np.array(Reff160)
tabfits['Re_f125_sig'] = np.array(Reff125_sig)
tabfits['Re_f160_sig'] = np.array(Reff160_sig)

tabfits['Rm_f125'] = np.array(Rmaj125)
tabfits['Rm_f160'] = np.array(Rmaj160)
tabfits['Rm_f125_sig'] = np.array(Rmaj125_sig)
tabfits['Rm_f160_sig'] = np.array(Rmaj160_sig)

tabfits['Ra_f125'] = np.array(Rarc125)
tabfits['Ra_f160'] = np.array(Rarc160)
tabfits['Ra_f125_sig'] = np.array(Rarc125_sig)
tabfits['Ra_f160_sig'] = np.array(Rarc160_sig)

#add sigma1

def Fphot(field, galaxy_id, phot):
    if phot.lower() == 'f125':
        bfilters = 203
    if phot.lower() == 'f160':
        bfilters = 205

    W, F, E, FLT = np.load(phot_path + '{0}_{1}_phot.npy'.format(field, galaxy_id))

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
        try:
            S1f125.append(Sigma_1('GSD', tabfits.id[i], 'f125', goodss_125))
            S1f160.append(Sigma_1('GSD', tabfits.id[i], 'f160', goodss_160))
        except:
            S1f125.append(-99)
            S1f160.append(-99)
    if tabfits.field[i][1] == 'N':
        try:
            S1f125.append(Sigma_1('GND', tabfits.id[i], 'f125', goodsn_125))
            S1f160.append(Sigma_1('GND', tabfits.id[i], 'f160', goodsn_160))
        except:
            S1f125.append(-99)
            S1f160.append(-99)
        
        
tabfits['Sigma1_f125'] = np.array(S1f125)
tabfits['Sigma1_f160'] = np.array(S1f160)

Re = []
Re_sig = []
Rm = []
Rm_sig = []
Ra = []
Ra_sig = []
n = []
n_f = []
n_sig = []
mag = []
mag_sig = []
S1 = []


for i in tabfits.index:
    if tabfits.zgrism[i] <= 1.5:
        Re.append(tabfits.Re_f125[i])
        Re_sig.append(tabfits.Re_f125_sig[i])
        Rm.append(tabfits.Rm_f125[i])
        Rm_sig.append(tabfits.Rm_f125_sig[i])
        Ra.append(tabfits.Ra_f125[i])
        Ra_sig.append(tabfits.Ra_f125_sig[i])
        n.append(tabfits.n_f125[i])
        n_f.append(tabfits.n_f125_f[i])
        n_sig.append(tabfits.n_f125_sig[i])
        mag.append(tabfits.mag_f125[i])
        mag_sig.append(tabfits.mag_f125_sig[i])
        S1.append(tabfits.Sigma1_f125[i])
        
    else:
        Re.append(tabfits.Re_f160[i])
        Re_sig.append(tabfits.Re_f160_sig[i])
        Rm.append(tabfits.Rm_f160[i])
        Rm_sig.append(tabfits.Rm_f160_sig[i])
        Ra.append(tabfits.Ra_f160[i])
        Ra_sig.append(tabfits.Ra_f160_sig[i])
        n.append(tabfits.n_f160[i])
        n_f.append(tabfits.n_f160_f[i])
        n_sig.append(tabfits.n_f160_sig[i])
        mag.append(tabfits.mag_f160[i])
        mag_sig.append(tabfits.mag_f160_sig[i])
        S1.append(tabfits.Sigma1_f160[i])

tabfits['Re'] = np.array(Re)
tabfits['Re_sig'] = np.array(Re_sig)
tabfits['Rm'] = np.array(Rm)
tabfits['Rm_sig'] = np.array(Rm_sig)
tabfits['Ra'] = np.array(Ra)
tabfits['Ra_sig'] = np.array(Ra_sig)
tabfits['n'] = np.array(n)
tabfits['n_f'] = np.array(n_f)
tabfits['n_sig'] = np.array(n_sig)
tabfits['mag'] = np.array(mag)
tabfits['mag_sig'] = np.array(mag_sig)
tabfits['Sigma1'] = np.array(S1)

from S1_err_tool import Extract_params, Sigma1_sig

dS1 = []
for i in tabfits.index:
#    if tabfits.zgrism[i] > 0:
    rshift = tabfits.zgrism[i]
#    else:
#        rshift = tabfits.zfit[i]

    try:
        re, n, Lg, Lp, M, dre, dn, dLg, dLp, dM = Extract_params(tabfits.field[i], tabfits.id[i], rshift,tabfits)
        dS1.append(Sigma1_sig(re, n, Lg, Lp, M, dre, dn, dLg, dLp, dM))
    except:
        dS1.append(-99)
        
tabfits['Sigma1_sig'] = np.array(dS1)
tabfits.to_pickle(pos_path + 'newfits_db.pkl')
