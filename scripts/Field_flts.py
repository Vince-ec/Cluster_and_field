from grizli import model
from grizli import multifit
from astropy.table import Table
from shutil import move
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from astropy.table import Table
from astropy import wcs
from astropy.io import fits
from glob import glob
import os

### set home for files
hpath = os.environ['HOME'] + '/'

def Match_field(B_ra, B_dec, R_ra, R_dec):
    r = (1. / 60 ) * 2
    in_out = np.repeat(False,len(R_ra))          
    for i in range(len(R_ra)):
        cosr = np.sin(np.radians(B_dec)) * np.sin(np.radians(R_dec[i])) + np.cos(np.radians(B_dec))\
                * np.cos(np.radians(R_dec[i])) * np.cos(np.radians(B_ra) - np.radians(R_ra[i]))
        rad = np.arccos(cosr)
        rad = np.degrees(rad)
        if rad < r:
            in_out[i]= True
    return in_out

def Field_select(field_name,g102_flts,g141_flts):  ### finds source in flt file, returns if present and the pos in pixels
    clear_flts = []
    clear_ra = []
    clear_dec = []
    
    threed_ra = []
    threed_dec = []
    threed_flts = []
    
    barro_ra = []
    barro_dec = []
    barro_flts = []
    
    for i in g102_flts:
        flt=fits.open(i)
        if flt[0].header['TARGNAME'] == field_name and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
            clear_ra.append(flt[0].header['RA_TARG'])
            clear_dec.append(flt[0].header['DEC_TARG'])
            clear_flts.append(i) 
      
    RA = np.mean(clear_ra)
    DEC = np.mean(clear_dec)
    
    for i in g141_flts:
        flt=fits.open(i)
        threed_ra.append(flt[0].header['RA_TARG'])
        threed_dec.append(flt[0].header['DEC_TARG'])
    
    in_out = Match_field(RA,DEC,threed_ra,threed_dec)
    threed_flts = np.array(g141_flts)[in_out]
    
    for i in g102_flts:
        flt=fits.open(i)
        barro_ra.append(flt[0].header['RA_TARG'])
        barro_dec.append(flt[0].header['DEC_TARG'])
    
    in_out = Match_field(RA,DEC,barro_ra,barro_dec)
    barro_flts = np.array(g102_flts)[in_out]
    
    in_out = np.repeat(True,len(barro_flts))
    
    for i in range(len(barro_flts)):
        if barro_flts[i] in clear_flts:
            in_out[i] = False
            
    barro_flts = barro_flts[in_out]

    clear_flts = np.append(clear_flts,barro_flts)
    return np.append(clear_flts,threed_flts)


class Extract_Grism_flts(object):
    def __init__(self, grism_flts, field,subfield):
        self.grism_flts = grism_flts
        self.field = field
        self.subfield = subfield
    
        if self.field == 'GSD':
            if hpath.strip('/Users/') == 'Vince.ec':
                self.mosaic = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sub.cat'
            if hpath.strip('/Users/') == 'vestrada':
                self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodss_v4.4/goodss-F105W-astrodrizzle-v4.4_drz_sub.cat'

            self.seg_map = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat',format='ascii').to_pandas()

        if self.field == 'GND':
            if hpath.strip('/Users/') == 'Vince.ec':
                self.mosaic = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = '/Volumes/Vince_research/Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sub.cat'
            if hpath.strip('/Users/') == 'vestrada':
                self.mosaic = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits'
                self.catalog = hpath + 'Data/CLEAR/CATALOGS/goodsn_v4.4/goodsn-F105W-astrodrizzle-v4.4_drz_sub.cat'

            self.seg_map = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
            self.ref_cat_loc = Table.read(hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.cat',format='ascii').to_pandas()
      
        self.ref_cat = Table.read(self.catalog,format='ascii')

 
        self.grp = multifit.GroupFLT(grism_files = self.grism_flts, direct_files = [], 
                      ref_file = self.mosaic,
                      seg_file = self.seg_map,
                      catalog = self.ref_cat,
                      cpu_count = 2,verbose = False)

        self.grp.compute_full_model(mag_limit=25, verbose=False)      
       
        self.grp.refine_list(poly_order=2, mag_limits=[16, 24], verbose=False)

        self.grp.save_full_data()

        if self.field[1] == 'N':
            flt_dir = 'n_flt_files'
        else:
            flt_dir = 's_flt_files'

        
        if not os.path.isdir('/Volumes/Vince_research/Data/Grism_fields/{0}'.format(self.subfield)):
            os.mkdir('/Volumes/Vince_research/Data/Grism_fields/{0}'.format(self.subfield))
        
        grismflts = glob('/Users/Vince.ec/Clear_data/{0}/*GrismFLT*'.format(flt_dir))
        grismwcs = glob('/Users/Vince.ec/Clear_data/{0}/*wcs.fits'.format(flt_dir))

        for i in grismflts:
            move(i,'/Volumes/Vince_research/Data/Grism_fields/{0}/{1}'.format(self.subfield,os.path.basename(i)))

        for i in grismwcs:
            move(i,'/Volumes/Vince_research/Data/Grism_fields/{0}/{1}'.format(self.subfield,os.path.basename(i)))



        grismflts = glob('/Users/Vince.ec/3dhst/{0}/*GrismFLT*'.format(flt_dir))
        grismwcs = glob('/Users/Vince.ec/3dhst/{0}/*wcs.fits'.format(flt_dir))

        for i in grismflts:
            move(i,'/Volumes/Vince_research/Data/Grism_fields/{0}/{1}'.format(self.subfield,os.path.basename(i)))

        for i in grismwcs:
            move(i,'/Volumes/Vince_research/Data/Grism_fields/{0}/{1}'.format(self.subfield,os.path.basename(i)))
            
            
###########################################################
#Beginning of script
###########################################################

Bflts = glob('/Users/Vince.ec/Clear_data/n_flt_files/*')
Rflts = glob('/Users/Vince.ec/3dhst/n_flt_files/*flt.fits')

subfield = 'GN1'
files = Field_select(subfield,Bflts,Rflts)

ex = Extract_Grism_flts(files,'GND',subfield)
