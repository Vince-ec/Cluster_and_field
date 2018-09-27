__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from spec_tools import Mag, Source_present, Scale_model
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import grizli
import pysynphot as S
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import os
from glob import glob
from grizli import model as griz_model
import collections
import pandas as pd
import re
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()

hpath = os.environ['HOME'] + '/'

def Gen_beam_fits(mosiac, seg_map, grism_data, catalog, gal_id, orient_id, loc, grism = 'G102'):
    if loc == 'south':
        pre = 'gs'
    if loc == 'north':
        pre = 'gn'
    if loc == 'uds':
        pre = 'uds'

    # initialize
    flt = griz_model.GrismFLT(grism_file = grism_data,
                          ref_file = mosiac, seg_file = seg_map,
                            pad=200, ref_ext=0, shrink_segimage=True,force_grism = grism)
    
    # catalog / semetation image
    ref_cat = Table.read( catalog ,format='ascii')
    seg_cat = flt.blot_catalog(ref_cat,sextractor=False)
    
    ## Reset
    flt.object_dispersers = collections.OrderedDict()

    flt.compute_full_model(ids=seg_cat['id'], mags=Mag(seg_cat['f_F125W']), mag_limit=Mag(seg_cat['f_F125W'][seg_cat['id'] == gal_id]) + 0.2 )

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
            
            co.write_fits(root='../beams/{0}_o{1}_{2}'.format(pre,orient,orient_id), clobber=True)
            fits.setval('../beams/{0}_o{1}_{2}_{3}.{4}.A.fits'.format(pre,orient, orient_id, gal_id,grism), 'EXPTIME', ext=0,
                        value=fits.open('../beams/{0}_o{1}_{2}_{3}.{4}.A.fits'.format(pre,orient, orient_id, gal_id,grism))[1].header['EXPTIME'])
        
    else:
        print('object not found')

def Gen_DB_and_beams(gid, loc, RA, DEC):
    if loc == 'south':
        g102_list = glob(hpath + 'Clear_data/s_flt_files/*flt.fits')
        g141_list = glob(hpath + '3dhst/s_flt_files/*flt.fits')
        ref = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F125W_orig_sci.fits'
        seg = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits'
        cat = hpath + 'Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat'
        pre = 'gs'

    if loc == 'north':
        g102_list = glob(hpath + 'Clear_data/n_flt_files/*flt.fits')
        g141_list = glob(hpath + '3dhst/n_flt_files/*flt.fits')    
        ref = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhst.v4.0.F125W_orig_sci.fits'
        seg = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhstP.seg.fits'
        cat = hpath + 'Clear_data/goodsn_mosaic/goodsn_3dhst.v4.3.cat'      
        pre = 'gn'

    if loc == 'uds':
        g102_list = glob(hpath + 'UDS_data/uds_g102_flts/*flt.fits')
        g141_list = glob(hpath + 'UDS_data/uds_g141_flts/*flt.fits')    
        ref = hpath + 'UDS_data/uds_mosaic/uds_3dhst.v4.0.F125W_orig_sci.fits'
        seg = hpath + 'UDS_data/uds_mosaic/uds_3dhst.v4.0.F160W_seg.fits'
        cat = hpath + 'UDS_data/uds_3dhst.v4.2.cats/Catalog/uds_3dhst.v4.2.cat'    
        pre = 'uds'
        
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

        g102_DB = g102_DB.sort_values('g102_orient')

        g141_DB = g141_DB.sort_values('g141_orient')
        
        g102_DB = g102_DB.reset_index().drop('index',axis=1)

        g141_DB = g141_DB.reset_index().drop('index',axis=1)
        
        obj_DB = pd.concat([g102_DB,g141_DB], ignore_index=True, axis=1)

        obj_DB.columns = ['g102_file','g102_orient','g102_xpos','g102_ypos','g141_file','g141_orient','g141_xpos','g141_ypos']

        obj_DB.to_pickle('../dataframes/file_list/{0}_{1}.pkl'.format(pre,gid))

        pa = obj_DB.g102_orient[0]
        num = 1

        if gid < 10000:
            galid = '0' + '{0}'.format(gid)
        else:
            galid = gid

        for i in obj_DB.index:
            if obj_DB.g102_orient[i] > 0:
                if pa  == obj_DB.g102_orient[i]:
                    if os.path.isfile('../beams/{0}_o{1}_{2}_{3}.g102.A.fits'.format(pre,int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num,loc)
                        pa  = obj_DB.g102_orient[i]
                        num += 1

                else:
                    pa  = obj_DB.g102_orient[i]
                    num = 1
                    if os.path.isfile('../beams/{0}_o{1}_{2}_{3}.g102.A.fits'.format(pre,int(pa), num, galid)):
                        num+=1
                    else:
                        Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num,loc)
                        num+=1

        pa = obj_DB.g141_orient[0]
        num = 1

        for i in obj_DB.index:
            if obj_DB.g141_orient[i] > 0:
                if pa  == obj_DB.g141_orient[i]:
                    if os.path.isfile('../beams/{0}_o{1}_{2}_{3}.g141.A.fits'.format(pre,int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num,loc, grism='G141')
                        pa  = obj_DB.g141_orient[i]
                        num += 1
                else:
                    pa  = obj_DB.g141_orient[i]
                    num = 1
                    if os.path.isfile('../beams/{0}_o{1}_{2}_{3}.g141.A.fits'.format(pre,int(pa), num, galid)):
                        num +=1
                    else:
                        Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num,loc, grism='G141')
                        num+=1
class Gen_spec(object):
    def __init__(self, gal_id, g102_min = 8700, g102_max = 11300, g141_min = 11100, g141_max = 16700, sim = True):
        self.gal_id = gal_id
        
        self.g102_list = glob('../beams/*{0}*g102*'.format(gal_id))
        self.g141_list = glob('../beams/*{0}*g141*'.format(gal_id))
        self.g102_wv, self.g102_fl, self.g102_er = self.Stack_1d_beams(self.g102_list,g102_min,g102_max) 
        self.g141_wv, self.g141_fl, self.g141_er = self.Stack_1d_beams(self.g141_list,g141_min,g141_max) 
        
        self.Stack_g102_g141()
        
        if sim == True:
            self.Initialize_sim()
            self.g102_sens = self.Set_sensitivity(self.g102_list[0],self.g102_wv)
            self.g141_sens = self.Set_sensitivity(self.g141_list[0],self.g141_wv)

    def Single_spec(self, beam, min_wv, max_wv):
        BEAM = griz_model.BeamCutout(fits_file= beam)
       
        ivar = BEAM.ivar
        weight = np.exp(-(1*np.abs(BEAM.contam)*np.sqrt(ivar)))
            
        w, f, e = BEAM.beam.optimal_extract(BEAM.grism.data['SCI'], bin=0, ivar=BEAM.ivar)

        flat = BEAM.flat_flam.reshape(BEAM.beam.sh_beam)
        fwave,fflux,ferr = BEAM.beam.optimal_extract(flat, bin=0, ivar=BEAM.ivar)
               
        f /= fflux
        e /= fflux

        IDX= [U for U in range(len(w)) if min_wv < w[U] < max_wv]
        
        return w[IDX], f[IDX], e[IDX]
        
    def Set_sensitivity(self,beam,master_wv):    
        BEAM = griz_model.BeamCutout(fits_file= beam)
        
        flat = BEAM.flat_flam.reshape(BEAM.beam.sh_beam)
        fwave,fflux,ferr = BEAM.beam.optimal_extract(flat, bin=0, ivar=BEAM.ivar)

        return interp1d(fwave,fflux)(master_wv)
        
    def Stack_spec(self, stk_wv, flgrid, errgrid):
        #### rearrange flux grid and generate weights
        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid) ## remove inif cause by nans in the error grid
        weigrid[infmask] = 0

        #### Stack spectra
        stack_fl, stack_er = np.zeros([2, len(stk_wv)])
        for i in range(len(stk_wv)):
            stack_fl[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            stack_er[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        
        return stk_wv, stack_fl, stack_er
        
    def Stack_1d_beams(self, beam_list, min_wv, max_wv):
        #### set master wavelength array
        wv,fl,er = self.Single_spec(beam_list[0], min_wv = min_wv, max_wv=max_wv)
        master_wv = wv[1:-1]
        
        #### intialize flux and error grid
        flgrid = np.zeros([len(beam_list), len(master_wv)])
        errgrid = np.zeros([len(beam_list), len(master_wv)])

        #### Get wv,fl,er for each spectra
        for i in range(len(beam_list)):
            wv,fl,er = self.Single_spec(beam_list[i], min_wv = min_wv, max_wv=max_wv)
            if sum(fl)>0:
                flgrid[i] = interp1d(wv, fl)(master_wv)
                errgrid[i] = interp1d(wv, er)(master_wv)
        
        return self.Stack_spec(master_wv, flgrid, errgrid)

    
    def Stack_g102_g141(self): #### good to display, but may not be good for science
        #### make combined wavelength set
        bounds = [min(self.g141_wv),max(self.g102_wv)]
        del_g102 = self.g102_wv[1] - self.g102_wv[0]
        del_g141 = self.g141_wv[1] - self.g141_wv[0]
        del_mix = (del_g102 + del_g141) / 2
        mix_wv = np.arange(bounds[0],bounds[1],del_mix)    
        stack_wv = np.append(np.append(self.g102_wv[self.g102_wv < bounds[0]],mix_wv),self.g141_wv[self.g141_wv > bounds[1]])

        #### intialize flux and error grid
        flgrid = np.zeros([2, len(stack_wv)])
        errgrid = np.zeros([2, len(stack_wv)])

        #### Get wv,fl,er for each spectra
        for i in range(len(stack_wv)):
            if min(self.g102_wv) <= stack_wv[i] <= max(self.g102_wv):
                flgrid[0][i] = interp1d(self.g102_wv, self.g102_fl)(stack_wv[i])
                errgrid[0][i] = interp1d(self.g102_wv, self.g102_er)(stack_wv[i])

            if min(self.g141_wv) <= stack_wv[i] <= max(self.g141_wv):
                flgrid[1][i] = interp1d(self.g141_wv, self.g141_fl)(stack_wv[i])
                errgrid[1][i] = interp1d(self.g141_wv, self.g141_er)(stack_wv[i])

        self.stack_wv, self.stack_fl, self.stack_er = self.Stack_spec(stack_wv, flgrid, errgrid)
        
    def Initialize_sim(self):
        #### pick out orients
        g102_beams = glob('../beams/*{0}*g102*'.format(self.gal_id))
        g102_beamid = [re.findall("o\w[0-9]+",U)[0] for U in g102_beams]
        self.g102_beamid = list(set(g102_beamid))

        g141_beams = glob('../beams/*{0}*g141*'.format(self.gal_id))
        g141_beamid = [re.findall("o\w[0-9]+",U)[0] for U in g141_beams]
        self.g141_beamid = list(set(g141_beamid))
        
        #### initialize dictionary of beams
        self.g102_beam_dict = {}
        self.g141_beam_dict = {}

        #### set beams for each orient
        for i in self.g102_beamid:
            key = i
            value = griz_model.BeamCutout(fits_file= glob('../beams/*{0}*{1}*g102*'.format(i,self.gal_id))[0])
            self.g102_beam_dict[key] = value 
            
        for i in self.g141_beamid:
            key = i
            value = griz_model.BeamCutout(fits_file= glob('../beams/*{0}*{1}*g141*'.format(i,self.gal_id))[0])
            self.g141_beam_dict[key] = value 
        
    def Sim_beam(self,BEAM, mwv, mfl, grism_wv, grism_fl, grism_er, grism_sens):
        ## Compute the models
        BEAM.beam.compute_model(spectrum_1d=[mwv, mfl], is_cgs = True)

        ## Extractions the model (error array here is meaningless)
        w, f, e = BEAM.beam.optimal_extract(BEAM.beam.model, bin=0)

        ## interpolate and scale
        f = interp1d(w,f)(grism_wv) / grism_sens
        C = Scale_model(grism_fl, grism_er,f)

        return C*f
            
    def Gen_sim(self, model_wv, model_fl, redshift): 
        ### normalize and redshift model spectra
        spec = S.ArraySpectrum(model_wv, model_fl, fluxunits='flam')
        spec = spec.redshift(redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')

        ### initialize model flux grids
        g102_mfl_grid = np.zeros([len(self.g102_beam_dict.keys()), len(self.g102_wv)])
        g141_mfl_grid = np.zeros([len(self.g141_beam_dict.keys()), len(self.g141_wv)])

        ### simulate each beam
        for i in range(len(self.g102_beamid)):    
            g102_mfl_grid[i] = self.Sim_beam(self.g102_beam_dict[self.g102_beamid[i]], spec.wave,spec.flux, 
                                             self.g102_wv, self.g102_fl, self.g102_er, self.g102_sens)

        for i in range(len(self.g141_beamid)):    
            g141_mfl_grid[i] = self.Sim_beam(self.g141_beam_dict[self.g141_beamid[i]], spec.wave,spec.flux, 
                                             self.g141_wv, self.g141_fl, self.g141_er, self.g141_sens)

        ### stack all sims
        self.g102_mfl = np.mean(g102_mfl_grid,axis=0)
        self.g141_mfl = np.mean(g141_mfl_grid,axis=0)