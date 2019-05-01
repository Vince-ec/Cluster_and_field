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
def:
-stack

class:
-Extract_all
--Extract_BeamCutout
--Phot_save
--Extract_spec
"""

def Stack(wv, fl , er, flt, line, cont):
    flgrid = np.transpose(fl)
    fltgrid = np.transpose(flt)
    linegrid = np.transpose(line)
    contgrid = np.transpose(cont)
    errgrid = np.transpose(er)
    weigrid = errgrid ** (-2)
    infmask = np.isinf(weigrid)
    weigrid[infmask] = 0
    ################

    stack, stack_flat, stack_line, stack_cont, err = np.zeros([5, len(wv[0])])
    for i in range(len(wv[0])):
        stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
        stack_flat[i] = np.sum(fltgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
        stack_line[i] = np.sum(linegrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
        stack_cont[i] = np.sum(contgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
        
        err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
    ################
    
    return wv[0], stack, err, stack_flat, stack_line, stack_cont

class Extract_all(object):
    def __init__(self, galaxy_id, field, grp_list) :
        self.galaxy_id = galaxy_id
        self.field = field
        self.grp = grp_list
    
        if self.field == 'GSD':
            self.ref_cat_loc = Table.read('/Volumes/Vince_CLEAR/CATALOGS/goodss_3dhst.v4.3.cat',format='ascii').to_pandas()

        if self.field == 'GND':
            self.ref_cat_loc = Table.read('/Volumes/Vince_CLEAR/CATALOGS/goodsn_3dhst.v4.3.cat',format='ascii').to_pandas()

                
    def Extract_BeamCutout(self):            
        beams = self.grp.get_beams(self.galaxy_id)

        pa = -1
        for i in beams:
            if i.grism.filter == 'G102':
                if pa != i.get_dispersion_PA():
                    pa = i.get_dispersion_PA()
                    i.write_fits(root='../Casey_data/beams/o{0}'.format(pa), clobber=True)
                    fits.setval('../Casey_data/beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
                            value=fits.open('../Casey_data/beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   

        pa = -1            
        for i in beams:
            if i.grism.filter == 'G141':
                if pa != i.get_dispersion_PA():
                    pa = i.get_dispersion_PA()
                    i.write_fits(root='../Casey_data/beams/o{0}'.format(pa), clobber=True)
                    fits.setval('../Casey_data/beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter), 'EXPTIME', ext=0,
                            value=fits.open('../Casey_data/beams/o{0}_{1}.{2}.A.fits'.format(pa, self.galaxy_id, i.grism.filter))[1].header['EXPTIME'])   

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
        Field = self.field[1]
        spec_list = glob('/Volumes/Vince_CLEAR/RELEASE_v2.0.0/*{0}*/*/Prep/*{1}*1D.fits'.format(Field, self.galaxy_id))

        Bwv, Bfl, Ber, Bft, Bln, Bct = [[],[],[],[],[],[]]

        Rwv, Rfl, Rer, Rft, Rln, Rct = [[],[],[],[],[],[]]

        for i in range(len(spec_list)):
            dat = fits.open(spec_list[i])

            try:
                Bwv.append(np.array(dat['G102'].data['wave']).T)
                Bfl.append(np.array(dat['G102'].data['flux']).T)
                Ber.append(np.array(dat['G102'].data['err']).T)
                Bft.append(np.array(dat['G102'].data['flat']).T)
                Bln.append(np.array(dat['G102'].data['line']).T)
                Bct.append(np.array(dat['G102'].data['cont']).T)

            except:
                print('no g102')

            try:
                Rwv.append(np.array(dat['G141'].data['wave']).T)
                Rfl.append(np.array(dat['G141'].data['flux']).T)
                Rer.append(np.array(dat['G141'].data['err']).T)
                Rft.append(np.array(dat['G141'].data['flat']).T)
                Rln.append(np.array(dat['G141'].data['line']).T)
                Rct.append(np.array(dat['G141'].data['cont']).T)

            except:
                print('no g141')
    
        if len(Bwv) > 0:                
            SBW, SBF, SBE, SBT, SBL, SBC = Stack(Bwv, Bfl, Ber, Bft, Bln, Bct)
            np.save('../spec_files/{0}_{1}_g102'.format(self.field, self.galaxy_id),[SBW, SBF, SBE, SBT, SBL, SBC])


        if len(Rwv) > 0:     
            SRW, SRF, SRE, SRT, SRL, SRC = Stack(Rwv, Rfl, Rer, Rft, Rln, Rct)
            np.save('../spec_files/{0}_{1}_g141'.format(self.field, self.galaxy_id),[SRW, SRF, SRE, SRT, SRL, SRC])

