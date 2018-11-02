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
from spec_tools import Source_present
from spec_tools import Photometry

### set home for files
hpath = os.environ['HOME'] + '/'

h=6.6260755E-27 # planck constant erg s
c=3E10          # speed of light cm s^-1
atocm=1E-8    # unit to convert angstrom to cm
kb=1.38E-16	    # erg k-1

"""
FUNCTIONS:

CLASSES:
-Extract_all
--Extract_BeamCutout
--Phot_save
--Extract_spec
"""

class Extract_all(object):
    def __init__(self, galaxy_id, field):
        self.galaxy_id = galaxy_id
        self.field = field
    
    
        if self.field == 'GSD':
            self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
            self.seg_map = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
            self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sub.cat'
            self.flt_path_g102 = hpath + 'Clear_data/s_flt_files/'
            self.flt_path_g141 = hpath + '3dhst/s_flt_files/'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat',format='ascii').to_pandas()

        if self.field == 'GND':
            self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
            self.seg_map = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
            self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sub.cat'
            self.flt_path_g102 = hpath + 'Clear_data/n_flt_files/'
            self.flt_path_g141 = hpath + '3dhst/n_flt_files/'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.cat',format='ascii').to_pandas()

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

    def Extract_BeamCutout(self, grism_file, instrument):
        flt = model.GrismFLT(grism_file = grism_file ,
                              ref_file = self.mosaic, seg_file = self.seg_map,
                                pad=200, ref_ext=0, shrink_segimage=True, force_grism = instrument)

        # catalog / semetation image
        if self.field == 'GSD':
            ref_cat = Table.read(hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat',format='ascii')        
        if self.field == 'GND':
            ref_cat = Table.read(hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhst.v4.3.cat',format='ascii') 
        
        seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
        flt.compute_full_model(ids=seg_cat['id'])
        beam = flt.object_dispersers[self.galaxy_id][2]['A']
        co = model.BeamCutout(flt, beam, conf=flt.conf)

        PA = np.round(fits.open(grism_file)[0].header['PA_V3'] , 1)

        co.write_fits(root='../beams/o{0}'.format(PA), clobber=True)

        ### add EXPTIME to extension 0


        fits.setval('../beams/o{0}_{1}.{2}.A.fits'.format(PA, self.galaxy_id, instrument), 'EXPTIME', ext=0,
                value=fits.open('../beams/o{0}_{1}.{2}.A.fits'.format(PA, self.galaxy_id, instrument))[1].header['EXPTIME'])   

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
        self.grp = multifit.GroupFLT(grism_files = self.grism_flts, direct_files = [], 
                      ref_file = self.mosaic,
                      seg_file = self.seg_map,
                      catalog = self.ref_cat,
                      cpu_count = 2,verbose = False)

        self.grp.compute_full_model(mag_limit=25)
        self.grp.refine_list(poly_order=2, mag_limits=[16, 24], verbose=False)

        beams = self.grp.get_beams(self.galaxy_id, size=80)
        self.mb = multifit.MultiBeam(beams, fcontam=0.0, group_name='../beams/{0}'.format(self.field))

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
        
        try:
            for i in g102_filt_list:
                if i in Pnum:
                    filter_102 = i
                    break

            for i in g141_filt_list:
                if i in Pnum:
                    filter_141 = i
                    break
        except:
            'No photometry for listed filters'

        pht1 = Photometry(Bwv[Bflx**2 > 0],Bflx[Bflx**2 > 0],Bflx[Bflx**2 > 0],filter_102)
        pht1.Get_Sensitivity()
        pht1.Photo_clipped()

        pht2 = Photometry(Rwv[Rflx**2 > 0],Rflx[Rflx**2 > 0],Rflx[Rflx**2 > 0],filter_141)
        pht2.Get_Sensitivity()
        pht2.Photo_clipped()

        Pwv, Pflx, Perr, Pnum = np.load('../phot/{0}_{1}_phot.npy'.format(self.field, self.galaxy_id))

        Bscale = Pflx[Pnum == filter_102] / pht1.photo
        Rscale = Pflx[Pnum == filter_141] / pht2.photo

        Bflx *= Bscale
        Berr *= Bscale
        Rflx *= Rscale
        Rerr *= Rscale

        np.save('../spec_files/{0}_{1}_g102'.format(self.field, self.galaxy_id),[Bwv, Bflx, Berr, Bflt])
        np.save('../spec_files/{0}_{1}_g141'.format(self.field, self.galaxy_id),[Rwv, Rflx, Rerr, Rflt])