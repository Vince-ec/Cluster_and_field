import numpy as np
import pandas as pd
import pickle
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy.table import Table
from spec_stats import Highest_density_region
from glob import glob
import os
import re
from spec_id import *
from spec_id_2d import *
### set home for files
hpath = os.environ['HOME'] + '/'


###### create tabit db#########
alldb = pd.read_pickle('all_galaxies.pkl')
#####parameters to get Z, m1-10, z, logmass, Av, lwa, z_50, t_50, z_q, t_q, log_ssfr, Reff, compactness

#make a dictionary
fitvals = {}
params = ['m', 'lm', 'z', 'd', 'lwa']
k = ['Z', 'lmass', 'zgrism', 'Av', 'lwa']

FIELD = alldb.field.values
GID = alldb.id.values


for p in range(len(params)):
    m = np.repeat(-99.0,len(FIELD))
    hci = []
    offmass = []
    
    for i in range(len(FIELD)):
        flist = glob(pos_path + '{}_{}_*_P{}.npy'.format(FIELD[i], GID[i],params[p]))
        try:
            flnm = 'none'
            for f in flist:
                ext = re.split('{}_{}_'.format(FIELD[i],GID[i]),
                re.split('_P{}.npy'.format(params[p]), os.path.basename(f))[0])[1]
                if ext in ['tabMfit', 'SFMfit']:
                    flnm = str(f)
                    break
            x,px = np.load(flnm)            
            m[i], mreg, oreg = Highest_density_region(px,x)
            print(m[i])
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
tabfits['field'] = alldb.field.values
tabfits['id'] = alldb.id.values
#tabfits['AGN'] = alldb.AGN.values

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

sfh_path = '/fdata/scratch/vestrada78840/SFH/'

for i in range(len(tabfits.index)):
    try:
    
        with open(sfh_path + '{}_{}.pkl'.format(tabfits.field[i], tabfits.id[i]), 'rb') as sfh_file:
            sfh = pickle.load(sfh_file)

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

mb = multifit.MultiBeam(beam_path + 'ERSPRIME_39170.beams.fits', **args)
Bwv = mb.oned_spectrum()['G102']['wave']
Bft = mb.oned_spectrum()['G102']['flat']

Rwv = mb.oned_spectrum()['G141']['wave']
Rft = mb.oned_spectrum()['G141']['flat']

iBft = interp1d(Bwv, Bft/ max(Bft), bounds_error=False, fill_value=0)
iRft = interp1d(Rwv, Rft/ max(Rft), bounds_error=False, fill_value=0)

def line_post_comb(fls, zfl, line):
    x,px = np.load(fls[0])
    y,py = np.load(fls[1])

    mx, hdrx, modx = Highest_density_region(px,x)
    my, hdry, mody = Highest_density_region(py,y)
    
    ###select mode
    lowlim = 1E-20
    hilim = 1E-14
    
    MODE = 'none'
    
    if (not lowlim < mx < hilim) | (not lowlim < my < hilim):
        MODE = 'filter'
    
    if MODE == 'none' and np.log10(np.abs(mx - my)/mx) > 1:
        MODE = 'autoselect'
        
    if MODE == 'none':
        MODE = 'combine'

    ###filter
    if MODE == 'filter':
        if lowlim < mx < hilim:
            return x,px
        
        if lowlim < my < hilim:
            return y,py
        
    ###autoselect by z
    if MODE == 'autoselect':
        z,pz = np.load(zfl)
        mz, hdrz, modz = Highest_density_region(pz,z)

        ln_nm = []
        ln_wv = []
        for k in args['t1']:
            if k[0] == 'l':
                ln_wv.append(np.round(args['t1'][k].wave[args['t1'][k].flux == max(args['t1'][k].flux)][0]))
                ln_nm.append(k[5:])
                
        Bllim = 7700
        Bhlim = 11300
        
        Rllim = 11300
        Rhlim = 16500
        
        LW = ln_wv[ln_nm == line]
        
        OLW = LW * (1+mz)
        
        if Bllim <= OLW <= Bhlim:
            return x,px

        if Rllim <= OLW <= Rhlim:
            return y,py 
                     
    ###combine
    if MODE == 'combine':
        z,pz = np.load(zfl)
        mz, hdrz, modz = Highest_density_region(pz,z)

        ln_nm = []
        ln_wv = []
        for k in args['t1']:
            if k[0] == 'l':
                ln_wv.append(np.round(args['t1'][k].wave[args['t1'][k].flux == max(args['t1'][k].flux)][0]))
                ln_nm.append(k[5:])
        
        LW = ln_wv[ln_nm == line]
        
        OLW = LW * (1+mz)

        allmin = [min(x),min(y)]
        allmax = [max(x),max(y)]

        X = np.linspace(min(allmin), max(allmax),1000)
        ipx = interp1d(x,px,bounds_error=False, fill_value=True)(X)
        ipy = interp1d(y,py,bounds_error=False, fill_value=True)(X)


        PX = ipx*iBft(OLW) + iRft(OLW) * ipy
        
        return X, PX / np.trapz(PX,X)

for k in args['t1']:
    if k[0] == 'l':
        m = np.repeat(-99.0,len(tabfits))
        hci = []
        offmass = []
        
        for i in range(len(tabfits.index)):
            flist = glob(pos_path + '{}_{}_*_Pline {}*.npy'.format(tabfits.field[i], tabfits.id[i],k[5:]))
            try:
                if len(flist) == 2:
                    zlist = glob(pos_path + '{}_{}_*_Pz.npy'.format(tabfits.field[i], tabfits.id[i]))
                    flnm = 'none'
                    for f in zlist:
                        ext = re.split('{}_{}_'.format(tabfits.field[i], tabfits.id[i]),
                        re.split('_Pz.npy', os.path.basename(f))[0])[1]
                        if ext in ['tabMfit', 'SFMfit']:
                            flnm = str(f)
                            break

                    z, pz = np.load(flnm)            
                    x, px = line_post_comb(flist, flnm, k[5:])

                    m[i], mreg, oreg = Highest_density_region(px,x)
                    hci.append(mreg)
                    offmass.append(oreg)

                else:
                    x,px = np.load(flist[0])            
                    m[i], mreg, oreg = Highest_density_region(px,x)
                    hci.append(mreg)
                    offmass.append(oreg)
            except:
                hci.append([0])
                offmass.append([0])
                
        tabfits[k[5:]] = m
        tabfits[k[5:] + '_hci'] = hci
        tabfits[k[5:] + '_modality'] = offmass


tabfits.to_pickle(pos_path + 'allfitsdb.pkl')