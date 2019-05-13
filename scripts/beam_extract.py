#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
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
import sys

if __name__ == '__main__':
    field = sys.argv[1] 
    subfield = sys.argv[2]



### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    data_path = '/fdata/scratch/vestrada78840/data/'
    model_path ='/fdata/scratch/vestrada78840/fsps_spec/'
    chi_path = '/fdata/scratch/vestrada78840/chidat/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'
    beam_path = '/fdata/scratch/vestrada78840/beams_v2/'
    template_path = '/fdata/scratch/vestrada78840/data/'
    out_path = '/fdata/scratch/vestrada78840/chidat/'
    pos_path = '/home/vestrada78840/posteriors/'
    phot_path = '/fdata/scratch/vestrada78840/phot/'

else:
    data_path = '../data/'
    model_path = hpath + 'fsps_models_for_fit/fsps_spec/'
    chi_path = '../chidat/'
    spec_path = '../spec_files/'
    beam_path = '../beams/'
    template_path = '../templates/'
    out_path = '../data/out_dict/'
    pos_path = '../data/posteriors/'
    phot_path = '../phot/'

class Extract_all(object):
    def __init__(self, galaxy_id, field, grp_list) :
        self.galaxy_id = galaxy_id
        self.field = field
        self.grp = grp_list
                  
    def Extract_BeamCutout(self):            
        beams = self.grp.get_beams(self.galaxy_id)

        #pa = -1
        #for i in beams:
        #    if i.grism.filter == 'G102':
        #        if pa != i.get_dispersion_PA():
        #            pa = i.get_dispersion_PA()
        #            i.write_fits(root = beam_path + 'o{0}'.format(pa), clobber=True)
        #            fits.setval(beam_path + 'o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
        #                    value=fits.open(beam_path + 'o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   

        pa = -1            
        for i in beams:
            if i.grism.filter == 'G141':
                if pa != i.get_dispersion_PA():
                    pa = i.get_dispersion_PA()
                    i.write_fits(root = beam_path + 'o{0}'.format(pa), clobber=True)
                    fits.setval(beam_path + 'o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
                            value=fits.open(beam_path + 'o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   
                    
##################################
Cpd = pd.read_pickle('/fdata/scratch/vestrada78840/Grism_flts/tabfitdb.pkl')

Cids = Cpd.query('field == "{0}"'.format(field)).id.values

grp = multifit.GroupFLT(grism_files = glob('/fdata/scratch/vestrada78840/Grism_flts/{0}/*GrismFLT.fits'.format(subfield)))
for ii in Cids:
    try:
        ex = Extract_all(ii, field, grp)
        ex.Extract_BeamCutout()
        ex=[]
    except:
        pass