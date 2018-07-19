import numpy as np
import pandas as pd
from astropy.io import fits
from grizli import model as griz_model
from scipy.interpolate import interp1d
from C_spec_id import Scale_model
import pysynphot as S
import re
import os
from glob import glob

class Gen_spec(object):
    def __init__(self, gal_id,field, g102_min = 8700, g102_max = 11400, g141_min = 11100, g141_max = 16700, sim = True):
        self.gal_id = gal_id
        self.field = field
        self.g102_list = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g102*'.format(field,gal_id))
        self.g141_list = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g141*'.format(field,gal_id))
        self.g102_wv, self.g102_fl, self.g102_er = self.Stack_1d_beams(self.g102_list,g102_min,g102_max) 
        self.g141_wv, self.g141_fl, self.g141_er = self.Stack_1d_beams(self.g141_list,g141_min,g141_max) 
        
        self.Stack_g102_g141()
        
        if sim == True:
            self.Initialize_sim()
            self.g102_sens = self.Set_sensitivity(self.g102_list[0],self.g102_wv)
            self.g141_sens = self.Set_sensitivity(self.g141_list[0],self.g141_wv)

    def Single_spec(self, beam, min_wv, max_wv):
        BEAM = griz_model.BeamCutout(fits_file= beam)

        w, f, e = BEAM.beam.optimal_extract(BEAM.grism.data['SCI'] - BEAM.contam, bin=0, ivar=BEAM.ivar)

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
        g102_beams = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g102*'.format(self.field,self.gal_id))
        g102_beamid = [re.findall("o\w[0-9]+",U)[0] for U in g102_beams]
        self.g102_beamid = list(set(g102_beamid))

        g141_beams = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g141*'.format(self.field,self.gal_id))
        g141_beamid = [re.findall("o\w[0-9]+",U)[0] for U in g141_beams]
        self.g141_beamid = list(set(g141_beamid))
        
        #### initialize dictionary of beams
        self.g102_beam_dict = {}
        self.g141_beam_dict = {}

        #### set beams for each orient
        for i in self.g102_beamid:
            key = i
            value = griz_model.BeamCutout(fits_file= glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*{2}*g102*'.format(self.field,i,self.gal_id))[0])
            self.g102_beam_dict[key] = value 
            
        for i in self.g141_beamid:
            key = i
            value = griz_model.BeamCutout(fits_file= glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*{2}*g141*'.format(self.field,i,self.gal_id))[0])
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
        
        
def Z_fit(gid, field, z, metal, age):
    g102_beams = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g102*'.format(self.field,self.gal_id))
    g141_beams = glob('/fdata/scratch/vestrada78840/beams/{0}*{1}*g141*'.format(self.field,self.gal_id))
    
    if (len(g102_beams) < 1)|(len(g141_beams) < 1):
        print('skipped')
    
    else:    
        ### define chi sqaure function for 3d grid
        def chi_sq_gr(data, error, model):
            return np.sum(((data - model) / error) ** 2, axis=1).reshape([z.size, metal.size, age.size]).astype(np.float128)

        ### read in model spectra
        spectra = np.load('/fdata/scratch/vestrada78840/data/z_fit_spectra.npy')    
        wv = spectra[0]
        fl = spectra[1:]

        ### create spectra object
        sp  = Gen_spec(gid, field, g102_max=11400, g141_min=11000)

        ### fit
        g102_mfl = np.zeros([z.size*metal.size*age.size,sp.g102_wv.size])
        g141_mfl = np.zeros([z.size*metal.size*age.size,sp.g141_wv.size])

        for i in range(z.size):
            for ii in range(metal.size):
                for iii in range(age.size):
                    sp.Gen_sim(wv,fl[ii*age.size + iii],z[i])
                    g102_mfl[i*metal.size*age.size + ii*age.size + iii]=sp.g102_mfl
                    g141_mfl[i*metal.size*age.size + ii*age.size + iii]=sp.g141_mfl

        ### gen chigrid
        chi102 = chi_sq_gr(sp.g102_fl,sp.g102_er,g102_mfl) 
        chi141 = chi_sq_gr(sp.g141_fl,sp.g141_er,g141_mfl)

        np.save('/home/vestrada78840/cluster_fit_data/chigrids/{0}{1}_g102_zfit'.format(field,gid),chi102) 
        np.save('/home/vestrada78840/cluster_fit_data/chigrids/{0}{1}_g141_zfit'.format(field,gid),chi141) 

        ####### Create normalize probablity
        P102 = np.exp(-chi102.astype(np.float128) / 2)
        P141 = np.exp(-chi141.astype(np.float128) / 2)

        prob102 = np.trapz(P102, age, axis=2)
        prob141 = np.trapz(P141, age, axis=2)

        C102 = np.trapz(np.trapz(prob102, metal, axis=1), z)
        C141 = np.trapz(np.trapz(prob141, metal, axis=1), z)

        prob102 /= C102
        prob141 /= C141

        #### Get z posteriors

        Pz102 = np.trapz(prob102, metal, axis=1)
        Pz141 = np.trapz(prob141, metal, axis=1)

        np.save('/home/vestrada78840/cluster_fit_data/posteriors/{0}{1}_g102_Pz'.format(field,gid),[z,Pz102]) 
        np.save('/home/vestrada78840/cluster_fit_data/posteriors/{0}{1}_g141_Pz'.format(field,gid),[z,Pz141])  