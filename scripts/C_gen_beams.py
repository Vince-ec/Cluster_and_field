import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
import grizli
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import os
from glob import glob
from grizli import model as griz_model
import collections
import pandas as pd

def Mag(band):
    magnitude=25-2.5*np.log10(band)
    return magnitude

def Source_present(fn,ra,dec):  ### finds source in flt file, returns if present and the pos in pixels
    flt=fits.open(fn)
    present = False
    
    w = wcs.WCS(flt[1].header)

    xpixlim=len(flt[1].data[0])
    ypixlim=len(flt[1].data)

    [pos]=w.wcs_world2pix([[ra,dec]],1)

    if -100 <pos[0]< xpixlim and 0 <pos[1]<ypixlim and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
        present=True
            
    return present,pos

def Gen_beam_fits(mosiac, seg_map, grism_data, catalog, gal_id, orient_id, grism = 'G102'):
    # initialize
    flt = griz_model.GrismFLT(grism_file = grism_data,
                          ref_file = mosiac, seg_file = seg_map,
                            pad=200, ref_ext=0, shrink_segimage=False,force_grism = grism)
    
    # catalog / semetation image
    ref_cat = Table.read( catalog ,format='ascii')
    seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
    
    ## Reset
    flt.object_dispersers = collections.OrderedDict()

    flt.compute_full_model(ids=seg_cat['id'], mags=-1)

    # check if galaxy is present
    if gal_id in flt.object_dispersers.keys():
    
        # reload(grizli.model)
        beam = flt.object_dispersers[gal_id][2]['A'] # can choose other orders if available
        beam.compute_model()

        ### BeamCutout object
        co = griz_model.BeamCutout(flt, beam, conf=flt.conf)

        ### Write the BeamCutout object to a normal FITS file
        orient = int(fits.open(grism_data)[0].header['PA_V3'])

        co.write_fits(root='../beams/o{0}_{1}'.format(orient,orient_id), clobber=True)
        fits.setval('../beams/o{0}_{1}_{2}.{3}.A.fits'.format(orient, orient_id, gal_id,grism), 'EXPTIME', ext=0,
                    value=fits.open('../beams/o{0}_{1}_{2}.{3}.A.fits'.format(orient, orient_id, gal_id,grism))[1].header['EXPTIME'])
        
    else:
        print('object not found')

def Gen_DB_and_beams(gid, loc, RA, DEC):
    if loc == 'south':
        g102_list = glob('/fdata/scratch/vestrada78840/gs_flt_files/*flt.fits')
        g141_list = glob('/fdata/scratch/vestrada78840/3ds_flt_files/*flt.fits')
        ref = '/fdata/scratch/vestrada78840/goodss_mosaic/goodss_3dhst.v4.0.F125W_orig_sci.fits'
        seg = '/fdata/scratch/vestrada78840/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
        cat = '/fdata/scratch/vestrada78840/goodss_mosaic/goodss_3dhst.v4.3.cat'

    if loc == 'north':
        g102_list = glob('/fdata/scratch/vestrada78840/gn_flt_files/*flt.fits')
        g141_list = glob('/fdata/scratch/vestrada78840/3dn_flt_files/*flt.fits')    
        ref = '/fdata/scratch/vestrada78840/goodsn_mosaic/goodsn_3dhst.v4.0.F125W_orig_sci.fits'
        seg = '/fdata/scratch/vestrada78840/goodsn_mosaic/goodsn_3dhstP.seg.fits'
        cat = '/fdata/scratch/vestrada78840/goodsn_mosaic/goodsn_3dhstP.cat'      
        
    flt_g102 = []
    obj_g102 =[]
    for i in range(len(g102_list)):
        pres,pos=Source_present(g102_list[i],RA,DEC)
        if pres==True:
            obj_g102.append(pos)
            flt_g102.append(g102_list[i])

    flt_g141 = []
    obj_g141 =[]
    for i in range(len(g141_list)):
        pres,pos=Source_present(g141_list[i],RA,DEC)
        if pres==True:
            obj_g141.append(pos)
            flt_g141.append(g141_list[i])
            
    g102_orients = []
    for i in range(len(flt_g102)):
        dat = fits.open(flt_g102[i])
        g102_orients.append(np.round(dat[0].header['PA_V3']).astype(int))

    g141_orients = []
    for i in range(len(flt_g141)):
        dat = fits.open(flt_g141[i])
        g141_orients.append(np.round(dat[0].header['PA_V3']).astype(int))
        
    xpos_g102,ypos_g102 = np.array(obj_g102).T
    xpos_g141,ypos_g141 = np.array(obj_g141).T

    g102_DB = pd.DataFrame({'g102_file' : flt_g102, 'g102_orient' : g102_orients, 'g102_xpos' : xpos_g102, 'g102_ypos' : ypos_g102})
    
    g141_DB = pd.DataFrame({'g141_file' : flt_g141, 'g141_orient' : g141_orients, 'g141_xpos' : xpos_g141, 'g141_ypos' : ypos_g141})
    
    obj_DB = pd.concat([g102_DB,g141_DB], ignore_index=True, axis=1)
    
    obj_DB.columns = ['g102_file','g102_orient','g102_xpos','g102_ypos','g141_file','g141_orient','g141_xpos','g141_ypos']
    
    obj_DB.to_pickle('../dataframes/file_list/{0}.pkl'.format(gid))

    pa = obj_DB.g102_orient[0]
    num = 1

    for i in obj_DB.index:
        if pa  == obj_DB.g102_orient[i]:
            Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
            pa  = obj_DB.g102_orient[i]
            num += 1

        else:
            pa  = obj_DB.g102_orient[i]
            num = 1
            Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
            num+=1

    pa = obj_DB.g141_orient[0]
    num = 1

    for i in obj_DB.index:
        if obj_DB.g141_orient[i] > 0:
            if pa  == obj_DB.g141_orient[i]:
                Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                pa  = obj_DB.g141_orient[i]
                num += 1

            else:
                pa  = obj_DB.g141_orient[i]
                num = 1
                Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                num+=1