__author__ = 'vestrada'

import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
from grizli import model
from grizli import multifit
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import os
import pandas as pd
from glob import glob
from spec_tools import Source_present, Photometry, Sig_int, Smooth, Scale_model

### set home for files
hpath = os.environ['HOME'] + '/'

h=6.6260755E-27 # planck constant erg s
c=3E10          # speed of light cm s^-1
atocm=1E-8    # unit to convert angstrom to cm
kb=1.38E-16	    # erg k-1

"""
FUNCTIONS:
-Box_phot
-Scale_spectra

CLASSES:
-Extract_all
--Extract_BeamCutout
--Phot_save
--Extract_spec
"""

def Box_phot(wv,fl,er,input_phot, phot_width = 150):
    
    h=6.6260755E-27 # planck constant erg s
    c=3E10          # speed of light cm s^-1
    atocm=1E-8  
    
    sens_wv = np.array([U for U in wv if (input_phot - phot_width) < U < (input_phot + phot_width)])
       
    wave = wv * atocm
    filtnu = c /(sens_wv * atocm)
    nu = c / wave
    fnu = (c / nu**2) * fl
    Fnu = interp1d(nu, fnu)(filtnu)
    ernu = (c/nu**2) * er
    Ernu = interp1d(nu, ernu)(filtnu)

    energy = 1 / (h *filtnu)

    top1 = Fnu * energy
    top = np.trapz(top1, filtnu)
    bottom1 = energy
    bottom = np.trapz(bottom1, filtnu)
    photonu = top / bottom

    tp = np.trapz(((np.log(sens_wv)) / sens_wv), sens_wv)
    bm = np.trapz(1 / sens_wv, sens_wv)

    wave_eff = np.exp(tp / bm)

    photo = photonu * (c / (wave_eff * atocm) ** 2)

    eff_wv = wave_eff
    photo = photo
    photo_er = Sig_int(filtnu, Ernu, np.ones(len(Ernu)), energy) * (c / (wave_eff * atocm) ** 2)
    
    return eff_wv, photo, photo_er

def Scale_spectra(bwv,bfl,ber,rwv,rfl,rer,pwv,pfl):
    Bp = []; Bs = []; Bsig = []
    Rp = [] ; Rs = []; Rsig = []

    IDb = [U for U in range(len(bwv)) if bfl[U]**2 > 0] 
    IDr = [U for U in range(len(rwv)) if rfl[U]**2 > 0] 
    
    for i in range(len(pwv)):
        if (min(bwv[IDb]) < pwv[i] - 150) and (max(bwv[IDb]) > pwv[i] + 150):
            eff_wv, phot, phot_er = Box_phot(bwv[IDb],Smooth(bfl[IDb],bwv[IDb]),ber[IDb], pwv[i])
            Bp.append(pfl[i]); Bs.append(phot); Bsig.append(phot_er)

        if (min(rwv[IDr]) < pwv[i] - 150) and (max(rwv[IDr]) > pwv[i] + 150):
            eff_wv, phot, phot_er = Box_phot(rwv[IDr],Smooth(rfl[IDr],rwv[IDr]),rer[IDr], pwv[i])
            Rp.append(pfl[i]); Rs.append(phot); Rsig.append(phot_er)

    bscale = Scale_model(np.array(Bs), np.array(Bsig), np.array(Bp))
    rscale = Scale_model(np.array(Rs), np.array(Rsig), np.array(Rp))    
    
    return bfl / bscale, ber / bscale, rfl / rscale, rer / rscale

class Extract_all(object):
    def __init__(self, galaxy_id, field, grp, cutout_size = 20):
        self.galaxy_id = galaxy_id
        self.field = field
        self.grp = grp
    
        if self.field == 'GSD':
            if hpath.strip('/Users/') == 'Vince.ec':
                self.mosaic = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sub.cat'
            if hpath.strip('/Users/') == 'vestrada':
                self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sub.cat'

            self.seg_map = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
            self.flt_path_g102 = hpath + 'Clear_data/s_flt_files/'
            self.flt_path_g141 = hpath + '3dhst/s_flt_files/'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat',format='ascii').to_pandas()

        if self.field == 'GND':
            if hpath.strip('/Users/') == 'Vince.ec':
                self.mosaic = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sub.cat'
            if hpath.strip('/Users/') == 'vestrada':
                self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sub.cat'

            self.seg_map = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
            self.flt_path_g102 = hpath + 'Clear_data/n_flt_files/'
            self.flt_path_g141 = hpath + '3dhst/n_flt_files/'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.cat',format='ascii').to_pandas()
        '''

        self.galaxy_ra = float(self.ref_cat_loc['ra'][self.ref_cat_loc['id'] == self.galaxy_id])
        self.galaxy_dec = float(self.ref_cat_loc['dec'][self.ref_cat_loc['id'] == self.galaxy_id])
        self.ref_cat = Table.read(self.catalog,format='ascii')

        flt_files = glob(self.flt_path_g102 + '*')

        self.grism_flts = []
        self.Bflts = []
        for i in flt_files:
            in_flt,loc = Source_present(i,self.galaxy_ra,self.galaxy_dec)
            if in_flt:
                self.grism_flts.append(i)
                self.Bflts.append(i)

        flt_files = glob(self.flt_path_g141 + '*')

        self.Rflts = []
        for i in flt_files:
            in_flt,loc = Source_present(i,self.galaxy_ra,self.galaxy_dec)
            if in_flt:
                self.grism_flts.append(i)
                self.Rflts.append(i)
        '''
        self.beams = self.grp.get_beams(self.galaxy_id, size = cutout_size)
                
    def Extract_BeamCutout(self, cutout_size = 20):
      
        pa = -1
        for i in self.beams:
            if i.grism.filter == 'G102':
                if pa != i.get_dispersion_PA():
                    pa = i.get_dispersion_PA()
                    i.write_fits(root='../beams/o{0}'.format(pa), clobber=True)
                    fits.setval('../beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
                            value=fits.open('../beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   
        
        pa = -1            
        for i in self.beams:
            if i.grism.filter == 'G141':
                if pa != i.get_dispersion_PA():
                    pa = i.get_dispersion_PA()
                    i.write_fits(root='../beams/o{0}'.format(pa), clobber=True)
                    fits.setval('../beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
                            value=fits.open('../beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   

    def Phot_save(self, masterlist = '../phot/master_template_list.pkl'):
        galdf = self.ref_cat_loc[self.ref_cat_loc.id == self.galaxy_id]
        master_tmp_df = pd.read_pickle(masterlist)

        if self.field == 'GSD':
            pre= 'S_'

        if self.field == 'GND':
            pre= 'N_'

        eff_wv = []
        phot_fl = []
        phot_er = []
        phot_num = []

        for i in galdf.keys():
            if i[0:2] == 'f_':
                Clam = 3E18 / master_tmp_df.eff_wv[master_tmp_df.tmp_name == pre + i].values[0] **2 * 10**((-1.1)/2.5-29)
                if galdf[i].values[0] > -99.0:
                    eff_wv.append(master_tmp_df.eff_wv[master_tmp_df.tmp_name == pre + i].values[0])
                    phot_fl.append(galdf[i].values[0]*Clam)
                    phot_num.append(master_tmp_df.tmp_num[master_tmp_df.tmp_name == pre + i].values[0])
            if i[0:2] == 'e_':
                if galdf[i].values[0] > -99.0:
                    phot_er.append(galdf[i].values[0]*Clam)

        np.save('../phot/{0}_{1}_phot'.format(self.field, self.galaxy_id), [eff_wv,phot_fl,phot_er,phot_num])

    def Extract_spec(self):
        self.mb = multifit.MultiBeam(self.beams, fcontam=1.0)

        g102 = self.mb.oned_spectrum()['G102']
        g141 = self.mb.oned_spectrum()['G141']

        Bwv = g102['wave']
        Bflx = g102['flux'] / g102['flat']
        Berr = g102['err'] / g102['flat']
        Bflt = g102['flat']

        Rwv = g141['wave']
        Rflx = g141['flux'] / g141['flat']
        Rerr = g141['err'] / g141['flat']
        Rflt = g141['flat']

        g102_filt_list = [201,240,202]
        g141_filt_list = [204,203,205]

        Pwv, Pflx, Perr, Pnum = np.load('../phot/{0}_{1}_phot.npy'.format(self.field, self.galaxy_id))
        
        Bflx2,Berr2,Rflx2,Rerr2 = Scale_spectra(Bwv, Bflx, Berr, Rwv, Rflx, Rerr, Pwv, Pflx)

        np.save('../spec_files/{0}_{1}_g102'.format(self.field, self.galaxy_id),[Bwv, Bflx2, Berr2, Bflt])
        np.save('../spec_files/{0}_{1}_g141'.format(self.field, self.galaxy_id),[Rwv, Rflx2, Rerr2, Rflt])
        

        
