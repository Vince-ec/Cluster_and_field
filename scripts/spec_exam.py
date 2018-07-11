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
        if obj_DB.g102_orient[i] > 0:
            if pa  == obj_DB.g102_orient[i]:
                if os.path.isfile('../beams/o{0}_{1}_{2}.g102.A.fits'.format(int(pa), num, gid)):
                    num +=1
                else:
                    Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
                    pa  = obj_DB.g102_orient[i]
                    num += 1

            else:
                pa  = obj_DB.g102_orient[i]
                num = 1
                if os.path.isfile('../beams/o{0}_{1}_{2}.g102.A.fits'.format(int(pa), num, gid)):
                    num+=1
                else:
                    Gen_beam_fits(ref,seg,obj_DB.g102_file[i],cat,gid,num)
                    num+=1

    pa = obj_DB.g141_orient[0]
    num = 1

    for i in obj_DB.index:
        if obj_DB.g141_orient[i] > 0:
            if pa  == obj_DB.g141_orient[i]:
                if os.path.isfile('../beams/o{0}_{1}_{2}.g141.A.fits'.format(int(pa), num, gid)):
                    num +=1
                else:
                    Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                    pa  = obj_DB.g141_orient[i]
                    num += 1
            else:
                pa  = obj_DB.g141_orient[i]
                num = 1
                if os.path.isfile('../beams/o{0}_{1}_{2}.g141.A.fits'.format(int(pa), num, gid)):
                    num +=1
                else:
                    Gen_beam_fits(ref, seg, obj_DB.g141_file[i], cat, gid, num, grism='G141')
                    num+=1
    
class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, pad=100, delayed = True,minwv = 7900, maxwv = 11300):
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.pad = pad
        self.delayed = delayed

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        if self.galaxy_id == 's35774':
            maxwv = 11100

        gal_wv, gal_fl, gal_er = np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S
        model = '../../../fsps_models_for_fit/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

    def Fit_lwa(self, fit_Z, fit_t, metal_array, age_array, tau_array):
        
        lwa_grid = np.load('../data/light_weight_scaling_3.npy')
        chi = []
        good_age =[]
        good_tau =[]
        for i in range(len(tau_array)):
            for ii in range(age_array.size):
                
                lwa = lwa_grid[np.argwhere(np.round(metal_array,3) == np.round(fit_Z,3))[0][0]][ii][i]
                
                if (fit_t - 0.1) < lwa < (fit_t + 0.1):
                    self.Sim_spec(fit_Z,age_array[ii],tau_array[i])
                    chi.append(sum(((self.gal_fl - self.fl) / self.gal_er)**2))
                    good_age.append(age_array[ii])
                    good_tau.append(tau_array[i])

        self.bfage = np.array(good_age)[chi == min(chi)][0]
        self.bftau = np.array(good_tau)[chi == min(chi)][0]
        if self.bftau == 0.0:
            self.bftau = int(0)
        self.Sim_spec(fit_Z, self.bfage, self.bftau)   


class Gen_sim(object):
    def __init__(self, galaxy_id, redshift, metal, age, tau, minwv=7900, maxwv=11400, pad=100):
        import pysynphot as S
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.metal = metal
        self.age = age
        self.tau = tau
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.gal_wv - output wavelength array of galaxy
        **
        self.gal_wv_rf - output wavelength array in restframe
        **
        self.gal_fl - output flux array of galaxy
        **
        self.gal_er - output error array of galaxy
        **
        self.fl - output flux array of model used for simulation
        **
        self.flx_err - output flux array of model perturb by the galaxy's 1 sigma errors
        **
        self.mfl - output flux array of model generated to fit against 
        """

        gal_wv, gal_fl, gal_er = np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0]
        self.gal_wv = self.gal_wv[self.gal_fl > 0]
        self.gal_er = self.gal_er[self.gal_fl > 0]
        self.gal_fl = self.gal_fl[self.gal_fl > 0]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

        ## create basis model for sim

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (self.metal, self.age, self.tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

    def Perturb_flux(self):
        self.flx_err = np.abs(self.fl + np.random.normal(0, self.gal_er))

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.mfl = C * adj_ifl


class Gen_spec_z(object):
    def __init__(self, spec_file, pad=100, minwv = 7900, maxwv = 11400):
        self.galaxy = spec_file
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """


        gal_wv, gal_fl, gal_er = np.load(self.galaxy)
        self.flt_input = '../data/galaxy_flts/n21156_flt.fits'

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + 1.251)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

    def Sim_spec(self, metal, age, redshift):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt8.0_spec.npy' % (metal, age)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

        
class Gen_spec_2d(object):
    def __init__(self, stack_2d, stack_2d_error, grism_flt, direct_flt, redshift):
        self.stack_2d = stack_2d
        self.stack_2d_error = stack_2d_error
        self.grism = grism_flt
        self.direct = direct_flt
        self.redshift = redshift

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        self.gal = np.load(self.stack_2d)
        self.err = np.load(self.stack_2d_error)
        
        flt = grizli.model.GrismFLT(grism_file= self.grism, 
                                direct_file= self.direct,
                                pad=200, ref_file=None, ref_ext=0, 
                                seg_file='../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits',
                                shrink_segimage=False)

        ref_cat = Table.read('../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat', format='ascii')
        sim_cat = flt.blot_catalog(ref_cat, sextractor=False)

        id = 39170

        x0 = ref_cat['x'][39169]+1
        y0 = ref_cat['y'][39169]+1

        mag =-2.5*np.log10(ref_cat['f_F850LP']) + 25
        keep = mag < 22

        flt.compute_full_model(ids=ref_cat['id'][keep],verbose=False, 
                               mags=mag[keep])

        ### Get the beams/orders
        beam = flt.object_dispersers[id]['A'] # can choose other orders if available
        beam.compute_model()

        ### BeamCutout object
        self.co = grizli.model.BeamCutout(flt, beam, conf=flt.conf)

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)
   
        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
  
        self.model = self.co.beam.compute_model(spectrum_1d=[spec.wave, spec.flux], 
                                           in_place=False).reshape(self.co.beam.sh_beam)

        adjmodel = np.append(np.zeros([4,len(self.model)]),self.model.T[:-4], axis=0).T
        
        rs = self.gal.shape[0]*self.gal.shape[1]
        C = Scale_model(self.gal.reshape(rs),self.err.reshape(rs),adjmodel.reshape(rs))
    
        self.sim = adjmodel*C