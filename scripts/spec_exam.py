__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from spec_tools import Mag, Source_present
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import grizli
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import os
from glob import glob
from grizli import model as griz_model
import collections
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()

hpath = os.environ['HOME'] + '/'

def Gen_beam_fits(mosiac, seg_map, grism_data, catalog, gal_id, orient_id, grism = 'G102'):
    # initialize
    flt = griz_model.GrismFLT(grism_file = grism_data,
                          ref_file = mosiac, seg_file = seg_map,
                            pad=200, ref_ext=0, shrink_segimage=True,force_grism = grism)
    
    # catalog / semetation image
    ref_cat = Table.read( catalog ,format='ascii')
    seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
    
    ## Reset
    flt.object_dispersers = collections.OrderedDict()

    flt.compute_full_model(ids=seg_cat['id'], mags=26)

    # check if galaxy is present
    if gal_id in flt.object_dispersers.keys():
    
        # reload(grizli.model)
        beam = flt.object_dispersers[gal_id][2]['A'] # can choose other orders if available
        beam.compute_model()
        
        # check if object in frame
        org = [beam.sly_parent.start,beam.slx_parent.start]
        nx = beam.slx_parent.stop - beam.slx_parent.start
        ny = beam.sly_parent.stop - beam.sly_parent.start
        
        if (org[1]<0) | (org[1]+nx > 1014 +2*200) | (org[0]<0) | (org[0]+ny > 1014 +2*200):
            print('object not found')
        
        else:
            ### BeamCutout object
            co = griz_model.BeamCutout(flt, beam, conf=flt.conf)

            ### Write the BeamCutout object to a normal FITS file
            orient = int(fits.open(grism_data)[0].header['PA_V3'])
            
            if gal_id < 10000:
                gal_id = '0' + '{0}'.format(gal_id)
            
            co.write_fits(root='../beams/o{0}_{1}'.format(orient,orient_id), clobber=True)
            fits.setval('../beams/o{0}_{1}_{2}.{3}.A.fits'.format(orient, orient_id, gal_id,grism), 'EXPTIME', ext=0,
                        value=fits.open('../beams/o{0}_{1}_{2}.{3}.A.fits'.format(orient, orient_id, gal_id,grism))[1].header['EXPTIME'])
        
    else:
        print('object not found')

def Gen_DB_and_beams(gid, loc, RA, DEC):
    if loc == 'south':
        g102_list = glob(hpath + 'Clear_data/s_flt_files/*flt.fits')
        g141_list = glob(hpath + '3dhst/s_flt_files/*flt.fits')
        ref = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F125W_orig_sci.fits'
        seg = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
        cat = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat'

    if loc == 'north':
        g102_list = glob(hpath + 'Clear_data/n_flt_files/*flt.fits')
        g141_list = glob(hpath + '3dhst/n_flt_files/*flt.fits')    
        ref = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhst.v4.0.F125W_orig_sci.fits'
        seg = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
        cat = hpath + 'uds_3dhst.v4.2.cats/Catalog/uds_3dhst.v4.2.cat'      
        
    if loc == 'uds':
        g102_list = glob(hpath + 'uds_flt_files/*flt.fits')
        g141_list = glob(hpath + '3dhst/n_flt_files/*flt.fits')    
        ref = hpath + ''
        seg = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
        cat = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.cat'    
        
        
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
        g102_orients.append(int(dat[0].header['PA_V3']))

    g141_orients = []
    for i in range(len(flt_g141)):
        dat = fits.open(flt_g141[i])
        g141_orients.append(int(dat[0].header['PA_V3']))
      
    if (len(obj_g102) < 1) | (len(obj_g141) < 1):
        print('object not found')
    else:
    
        xpos_g102,ypos_g102 = np.array(obj_g102).T
        xpos_g141,ypos_g141 = np.array(obj_g141).T

        g102_DB = pd.DataFrame({'g102_file' : flt_g102, 'g102_orient' : g102_orients, 'g102_xpos' : xpos_g102, 'g102_ypos' : ypos_g102})

        g141_DB = pd.DataFrame({'g141_file' : flt_g141, 'g141_orient' : g141_orients, 'g141_xpos' : xpos_g141, 'g141_ypos' : ypos_g141})

        obj_DB = pd.concat([g102_DB,g141_DB], ignore_index=True, axis=1)

        obj_DB.columns = ['g102_file','g102_orient','g102_xpos','g102_ypos','g141_file','g141_orient','g141_xpos','g141_ypos']

        obj_DB.to_pickle('../dataframes/file_list/{0}.pkl'.format(gid))

        pa = obj_DB.g102_orient[0]
        num = 1

        if gid < 10000:
            galid = '0' + '{0}'.format(gid)
        else:
            galid = gid

        for i in obj_DB.index:
            if obj_DB.g102_orient[i] > 0:
                if pa  == obj_DB.g102_orient[i]:
                    if os.path.isfile('../beams/o{0}_{1}_{2}.g102.A.fits'.format(int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
                        pa  = obj_DB.g102_orient[i]
                        num += 1

                else:
                    pa  = obj_DB.g102_orient[i]
                    num = 1
                    if os.path.isfile('../beams/o{0}_{1}_{2}.g102.A.fits'.format(int(pa), num, galid)):
                        num+=1
                    else:
                        Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
                        num+=1

        pa = obj_DB.g141_orient[0]
        num = 1

        for i in obj_DB.index:
            if obj_DB.g141_orient[i] > 0:
                if pa  == obj_DB.g141_orient[i]:
                    if os.path.isfile('../beams/o{0}_{1}_{2}.g141.A.fits'.format(int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                        pa  = obj_DB.g141_orient[i]
                        num += 1
                else:
                    pa  = obj_DB.g141_orient[i]
                    num = 1
                    if os.path.isfile('../beams/o{0}_{1}_{2}.g141.A.fits'.format(int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                        num+=1
    
    
class Single_spec(object):
    def __init__(self, beam, gal_id,min_wv = 8000, max_wv = 11500):
        BEAM = griz_model.BeamCutout(fits_file= beam)

        self.spec_2D = BEAM.grism.data['SCI']
        self.contam_2D = BEAM.contam
        self.clean_2D = BEAM.grism.data['SCI'] - BEAM.contam
        self.cutout = BEAM.beam.direct*(BEAM.beam.seg == gal_id)

        xspec, yspec, yerr = BEAM.beam.optimal_extract(BEAM.grism.data['SCI'], bin=0, ivar=BEAM.ivar) #data
        
        flat = BEAM.flat_flam.reshape(BEAM.beam.sh_beam)
        fwave,fflux,ferr = BEAM.beam.optimal_extract(flat, bin=0, ivar=BEAM.ivar)
                
        yspec /= fflux
        yerr /=fflux

        IDX= [U for U in range(len(xspec)) if min_wv < xspec[U] < max_wv]
        
        self.wv = xspec[IDX]
        self.fl = yspec[IDX]
        self.er = yerr[IDX]

        
class Stack_spec(object):
    def __init__(self, gal_id, g102_min = 8700, g102_max = 11300, g141_min = 11100, g141_max = 16450):
        self.gal_id = gal_id
        p_file = glob('../dataframes/phot/*{0}*'.format(gal_id))[0]
        self.phot_db = pd.read_pickle(p_file)
        
        self.g102_list = glob('../beams/*{0}*g102*'.format(gal_id))
        self.g141_list = glob('../beams/*{0}*g141*'.format(gal_id))
        self.g102_wv, self.g102_fl, self.g102_er = self.Stack_1d_beams(self.g102_list,g102_min,g102_max) 
        self.g141_wv, self.g141_fl, self.g141_er = self.Stack_1d_beams(self.g141_list,g141_min,g141_max) 
        
        self.Stack_g102_g141()
        
    def Stack_1d_beams(self, beam_list, min_wv, max_wv):
        spec = Single_spec(beam_list[0], self.gal_id, min_wv = min_wv, max_wv=max_wv)

        stack_wv = spec.wv[1:-1]

        flgrid = np.zeros([len(beam_list), len(stack_wv)])
        errgrid = np.zeros([len(beam_list), len(stack_wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(beam_list)):
            spec = Single_spec(beam_list[i], self.gal_id, min_wv = min_wv, max_wv=max_wv)
            flgrid[i] = interp1d(spec.wv, spec.fl)(stack_wv)
            errgrid[i] = interp1d(spec.wv, spec.er)(stack_wv)
        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(stack_wv)])
        for i in range(len(stack_wv)):
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        stack_fl = np.array(stack)
        stack_er = np.array(err)

        return stack_wv, stack_fl, stack_er
    
    def Stack_g102_g141(self):
        
        bounds = [min(self.g141_wv),max(self.g102_wv)]
        del_g102 = self.g102_wv[1] - self.g102_wv[0]
        del_g141 = self.g141_wv[1] - self.g141_wv[0]
        del_mix = (del_g102 + del_g141) / 2
        mix_wv = np.arange(bounds[0],bounds[1],del_mix)    
        stack_wv = np.append(np.append(self.g102_wv[self.g102_wv < bounds[0]],mix_wv),self.g141_wv[self.g141_wv > bounds[1]])

        flgrid = np.zeros([2, len(stack_wv)])
        errgrid = np.zeros([2, len(stack_wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(stack_wv)):
            if min(self.g102_wv) <= stack_wv[i] <= max(self.g102_wv):
                flgrid[0][i] = interp1d(self.g102_wv, self.g102_fl)(stack_wv[i])
                errgrid[0][i] = interp1d(self.g102_wv, self.g102_er)(stack_wv[i])

            if min(self.g141_wv) <= stack_wv[i] <= max(self.g141_wv):
                flgrid[1][i] = interp1d(self.g141_wv, self.g141_fl)(stack_wv[i])
                errgrid[1][i] = interp1d(self.g141_wv, self.g141_er)(stack_wv[i])
        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(stack_wv)])
        for i in range(len(stack_wv)):
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.stack_wv = stack_wv
        self.stack_fl = np.array(stack)
        self.stack_er = np.array(err)