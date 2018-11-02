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
-Extract_BeamCutout
-Phot_save
-Extract_spec

CLASSES:
"""
def Extract_BeamCutout(target_id, grism_file, mosaic, seg_map, instruement, catalog):
    flt = model.GrismFLT(grism_file = grism_file ,
                          ref_file = mosaic, seg_file = seg_map,
                            pad=200, ref_ext=0, shrink_segimage=True, force_grism = instrument)
    
    # catalog / semetation image
    ref_cat = Table.read(catalog ,format='ascii')
    seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
    flt.compute_full_model(ids=seg_cat['id'])
    beam = flt.object_dispersers[target_id][2]['A']
    co = model.BeamCutout(flt, beam, conf=flt.conf)
    
    PA = np.round(fits.open(grism_file)[0].header['PA_V3'] , 1)
    
    co.write_fits(root='beams/o{0}'.format(PA), clobber=True)

    ### add EXPTIME to extension 0
    
    
    fits.setval('../beams/o{0}_{1}.{2}.A.fits'.format(PA, target_id, instrument), 'EXPTIME', ext=0,
            value=fits.open('../beams/o{0}_{1}.{2}.A.fits'.format(PA, target_id, instrument))[1].header['EXPTIME'])   

def Phot_save(catalog_DF, galaxy_id, field, masterlist = '../phot/master_template_list.pkl'):
    galdf = catalog_DF[catalog_DF.id == galaxy_id]
    master_tmp_df = pd.read_pickle(masterlist)
        
    if field == 'GSD':
        pre= 'S_'
     
    if field == 'GND':
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

    np.save('../phot/{0}_{1}_phot'.format(field, galaxy_id), [eff_wv,phot_fl,phot_er,phot_num])
    
def Extract_spec(mosaic, seg_map, catalog, ref_cat_loc, g102_flts, g141_flts, 
                 field, galaxy_id, filter_102 = 201, filter_141 = 203):

    ref_cat = Table.read(catalog,format='ascii')

    galaxy_ra = float(ref_cat_loc['ra'][ref_cat_loc['id'] == galaxy_id])
    galaxy_dec = float(ref_cat_loc['dec'][ref_cat_loc['id'] == galaxy_id])

    flt_files = glob(g102_flts + '*')

    grism_flts = []
    for i in flt_files:
        in_flt,loc = Source_present(i,galaxy_ra,galaxy_dec)
        if in_flt:
            grism_flts.append(i)
            print('x={0:0.1f} y={1:0.1f}, PA={2:0.1f}, file={3} '.format(
                loc[0],loc[1],fits.open(i)[0].header['PA_V3'], os.path.basename(i)))
        
    flt_files = glob(g141_flts + '*')

    for i in flt_files:
        in_flt,loc = Source_present(i,galaxy_ra,galaxy_dec)
        if in_flt:
            grism_flts.append(i)
            print('x={0:0.1f} y={1:0.1f}, PA={2:0.1f}, file={3} '.format(
                loc[0],loc[1],fits.open(i)[0].header['PA_V3'], os.path.basename(i)))
        
    grp = multifit.GroupFLT(grism_files = grism_flts, direct_files = [], 
                  ref_file = mosaic,
                  seg_file = seg_map,
                  catalog = ref_cat,
                  cpu_count = 1,verbose = False)

    grp.compute_full_model(mag_limit=25)
    grp.refine_list(poly_order=2, mag_limits=[16, 24], verbose=False)

    beams = grp.get_beams(galaxy_id, size=80)
    mb = multifit.MultiBeam(beams, fcontam=0.0, group_name='../beams/{0}'.format(field))

    g102 = mb.oned_spectrum()['G102']
    g141 = mb.oned_spectrum()['G141']

    Bwv = g102['wave']
    Bflx = g102['flux'] / g102['flat']
    Berr = g102['err'] / g102['flat']
    Bflt = g102['flat']
    
    Rwv = g141['wave']
    Rflx = g141['flux'] / g141['flat']
    Rerr = g141['err'] / g141['flat']
    Rflt = g141['flat']
      
    pht1 = Photometry(Bwv,Bflx,Bflx,filter_102)
    pht1.Get_Sensitivity()
    pht1.Photo()

    pht2 = Photometry(Rwv,Rflx,Rflx,filter_141)
    pht2.Get_Sensitivity()
    pht2.Photo()
        
    Pwv, Pflx, Perr, Pnum = np.load('../phot/{0}_{1}_phot.npy'.format(field, galaxy_id))
        
    Bscale = Pflx[Pnum == filter_102] / pht1.photo
    Rscale = Pflx[Pnum == filter_141] / pht2.photo
    
    Bflx *= Bscale
    Berr *= Bscale
    Rflx *= Rscale
    Rerr *= Rscale
    
    np.save('../spec_files/{0}_{1}_g102'.format(field, galaxy_id),[Bwv, Bflx, Berr, Bflt])
    np.save('../spec_files/{0}_{1}_g141'.format(field, galaxy_id),[Rwv, Rflx, Rerr, Rflt])