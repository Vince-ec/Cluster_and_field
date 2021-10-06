import numpy as np
import pandas as pd
import os
from glob import glob
from scipy.interpolate import interp1d, interp2d

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sea

from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from sim_engine import Scale_model
from spec_tools import Source_present, Oldest_galaxy, Sig_int, Smooth
from spec_stats import Smooth, Highest_density_region
from spec_id import *
from spec_exam import Gen_spec

from grizli import multifit
from grizli import model
from grizli.utils import SpectrumTemplate


sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)
### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/Users/Vince.ec/':
    dpath = '/Volumes/Vince_research/Data/' 
    
else:
    dpath = hpath + 'Data/' 

def Gen_initial_MB(field, gid):
    # get beam list
    fl = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/BEAMS/*{}*/*{}*'.format(field,gid))
    
    # sort beams
    sz = []
    for f in fl:
        sz.append(os.path.getsize(f))

    fl = np.array(fl)[np.argsort(sz)]

    # remove repeats
    nlist = []
    blist = []
    for f in fl:
        mb = multifit.MultiBeam(f,**args)
        for bm in mb.beams:
            if bm.grism.parent_file not in nlist:
                nlist.append(bm.grism.parent_file)
                blist.append(bm)
    
    #make the mb
    mb = multifit.MultiBeam(blist,**args)

    for b in mb.beams:
        if hasattr(b, 'xp'):
            delattr(b, 'xp')
    mb.initialize_masked_arrays()
    
    return mb

def Plot_grism(MB,ax, color,instr, lims,z):
    sptbl = MB.oned_spectrum()

    w = sptbl[instr]['wave']
    f = sptbl[instr]['flux']
    e = sptbl[instr]['err']
    fl = sptbl[instr]['flat']
        
    clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1]]

    ax.errorbar(w[clip]/(1+z),f[clip]/fl[clip],e[clip]/fl[clip], color = color,
                linestyle='None', marker='o', markersize=0.25, zorder = 1, elinewidth = 1)

def Plot_beams(mb,W,P,E, field, gid,z):
    gs = gridspec.GridSpec(2,2)

    plt.figure(figsize=[20,12])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])

    for bm in mb.beams:
        xspec, yspec, yerr = bm.beam.optimal_extract(bm.grism.data['SCI'] - bm.contam,ivar = bm.ivar)

        flat_model = bm.flat_flam.reshape(bm.beam.sh_beam)
        xspecm, yspecm, yerrm = bm.beam.optimal_extract(flat_model)

        if bm.grism.filter == 'G102':
            IDX = [U for U in range(len(xspec)) if 8000 < xspec[U] < 11300]
            ax1.plot(xspec[IDX]/(1+z),yspec[IDX]/yspecm[IDX], color = 'b')
        else:
            IDX = [U for U in range(len(xspec)) if 11200 < xspec[U] < 16500]
            ax2.plot(xspec[IDX]/(1+z),yspec[IDX]/yspecm[IDX], color = 'r') 
    
    ax = plt.subplot(gs[1,:])
    try:
        Plot_grism(mb , ax, 'b', 'G102', [8000,11500],z)
    except:
        pass
    try:
        Plot_grism(mb , ax, 'r', 'G141', [11000,16500],z)
    except:
        pass
    
    IDP = [U for U in range(len(W)) if P[U]/E[U]  > 0.01]

    
    ax.errorbar(np.array(W)[IDP]/(1+z),np.array(P)[IDP],np.array(E)[IDP], fmt='o', color='k', zorder=0)
    ax.set_xscale('log')  
    plt.title('z={}'.format(z))
    plt.savefig('../plots/newspec_exam_2/G{}D-{}_beams.png'.format(field, gid),bbox_inches = 'tight')
    
def Beam_cleanup(mb, B_condition=[], R_condition=[]):
    ## conditions in form of [low-wv, hi-wv, gtr or less, flux, clip or omit]
    Bselect = False
    Rselect = False
    
    if len(B_condition) == 5:
        Bwvl, Bwvh, B_cond, Bfl, B_cl = B_condition
        Bselect = True

    if len(R_condition) == 5:
        Rwvl, Rwvh, R_cond, Rfl, R_cl = R_condition
        Rselect = True

    BEAM_exempt =[]
    ids = 0
    cleanspec = []
    clip_lims = []
    ### selection for bad beams
    for bm in mb.beams:
        xspec, yspec, yerr = bm.beam.optimal_extract(bm.grism.data['SCI'] - bm.contam,ivar = bm.ivar)
        flat_model = bm.flat_flam.reshape(bm.beam.sh_beam)
        xspecm, yspecm, yerrm = bm.beam.optimal_extract(flat_model)
        bex = False

        if Bselect:
            for i in range(len(xspec)):
                if B_cond == 'gtr':
                    if (bm.grism.filter == 'G102' and Bwvl < xspec[i] < Bwvh) and (yspec[i]/yspecm[i]) > Bfl:
                        bex = True
                        cleanspec.append(B_cl)
                        clip_lims.append([Bwvl, Bwvh])
                        break

                if B_cond == 'less':
                    if (bm.grism.filter == 'G102' and Bwvl < xspec[i] < Bwvh) and (yspec[i]/yspecm[i]) < Bfl:
                        bex = True
                        cleanspec.append(B_cl)
                        clip_lims.append([Bwvl, Bwvh])
                        break
        
        if Rselect:
            for i in range(len(xspec)):
                if R_cond == 'gtr':
                    if (bm.grism.filter == 'G141' and Rwvl < xspec[i] < Rwvh) and (yspec[i]/yspecm[i]) > Rfl:
                        bex = True
                        cleanspec.append(R_cl)
                        clip_lims.append([Rwvl, Rwvh])
                        break

                if R_cond == 'less':
                    if (bm.grism.filter == 'G141' and Rwvl < xspec[i] < Rwvh) and (yspec[i]/yspecm[i]) < Rfl:
                        bex = True
                        cleanspec.append(R_cl)
                        clip_lims.append([Rwvl, Rwvh])
                        break

        if bex:
            BEAM_exempt.append(bm.grism.parent_file)

    ### set up selection settings
    
    omitspec = np.zeros(len(BEAM_exempt))
    clipspec = np.zeros(len(BEAM_exempt))
    for i in range(len(cleanspec)):
        if cleanspec[i] == 'clip':
            clipspec[i] = 1
            
        if cleanspec[i] == 'omit':
            omitspec[i] = 1   
    return BEAM_exempt, clip_lims, clipspec, omitspec
            
def Clean_mb(mb, BEAM_exempt, clip_lims, clipspec, omitspec):   
    fblist = []
    idc = 0
    for bm in mb.beams:
        if bm.grism.parent_file in BEAM_exempt:            
            if clipspec[idc] == 1:
                xspec, yspec, yerr = bm.beam.optimal_extract(bm.grism.data['SCI'] - bm.contam,ivar = bm.ivar) 
                lms = clip_lims[idc]
                for i in range(len(xspec)):
                    if lms[0] < xspec[i]< lms[1]:
                        bm.grism.data['SCI'].T[i] = np.zeros_like(bm.grism.data['SCI'].T[i])
                        bm.grism.data['ERR'].T[i] = np.ones_like(bm.grism.data['ERR'].T[i])*1000  

            if omitspec[idc] == 1:
                pass
            else:    
                fblist.append(bm)

            idc += 1

        else:    
            fblist.append(bm)   

    mb = multifit.MultiBeam(fblist,**args)
    for b in mb.beams:
        if hasattr(b, 'xp'):
            delattr(b, 'xp')
    mb.initialize_masked_arrays()
    
    return mb

def Phot_load(field, galaxy_id,ref_cat_loc,masterlist = '../phot/master_template_list.pkl'):
    galdf = ref_cat_loc[ref_cat_loc.id == galaxy_id]
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
    
    return eff_wv, phot_fl, phot_er
       
########################################################################################
########################################################################################
    
# CNdb = pd.read_pickle('../dataframes/galaxy_frames/massMetal_GND_full.pkl')
# CSdb = pd.read_pickle('../dataframes/galaxy_frames/massMetal_GSD_full.pkl')

NGSID = np.load('../dataframes/N_GSD.npy', allow_pickle=True)
NGNID = np.load('../dataframes/N_GND.npy', allow_pickle=True)

NGSz = np.load('../dataframes/N_GSD_z.npy', allow_pickle=True)
NGNz = np.load('../dataframes/N_GND_z.npy', allow_pickle=True)


v4Ncat = Table.read(hpath + 'Downloads/goodsn_3dhst.v4.5.cat',
                 format='ascii').to_pandas()
v4Scat = Table.read(hpath + 'Downloads/goodss_3dhst.v4.5.cat',
                 format='ascii').to_pandas()

temps = {}
for k in args['t1']:
    if k[0] == 'f' or k[5:] in ['Ha', 'Hb', 'Hg', 'Hd'] :
        temps[k] = args['t1'][k]

field = 'S'
cat = v4Scat 
# db = CSdb

for idx in range(len(NGSID)):
    rshift = NGSID[idx] 

    if len(str(NGSID[idx])) < 5:
        gid = '0' + str(NGSID[idx])
    else:
        gid = str(NGSID[idx])

    if not os.path.isfile('../beams/beam_config/GSD_{}.npy'.format(gid)):
        print('run',glob('../beams/beam_config/GSD_{}.npy'.format(gid)))
        W,P,E = Phot_load('G{}D'.format(field), NGSID[idx], cat)

        mb  = Gen_initial_MB(field, gid)

        Plot_beams(mb,W,P,E, field, gid,NGSz[idx])
        BMX, Clims, Cspec, Ospec = Beam_cleanup(mb, B_condition=[], R_condition=[])

        np.save('../beams/beam_config/G{}D_{}_ex'.format(field, gid),[BMX])
        np.save('../beams/beam_config/G{}D_{}'.format(field, gid),[Clims, Cspec, Ospec])

    else:
        print(glob('../beams/beam_config/GSD_{}.npy'.format(gid)))

field = 'N'
cat = v4Ncat 
# db = CSdb

for idx in range(len(NGNID)):
    rshift = NGNID[idx] 

    if len(str(NGNID[idx])) < 5:
        gid = '0' + str(NGNID[idx])
    else:
        gid = str(NGNID[idx])

    if not os.path.isfile('../beams/beam_config/GND_{}.npy'.format(gid)):
        print('run',glob('../beams/beam_config/GND_{}.npy'.format(gid)))
        W,P,E = Phot_load('G{}D'.format(field), NGNID[idx], cat)

        mb  = Gen_initial_MB(field, gid)

        Plot_beams(mb,W,P,E, field, gid,NGNz[idx])
        BMX, Clims, Cspec, Ospec = Beam_cleanup(mb, B_condition=[], R_condition=[])

        np.save('../beams/beam_config/G{}D_{}_ex'.format(field, gid),[BMX])
        np.save('../beams/beam_config/G{}D_{}'.format(field, gid),[Clims, Cspec, Ospec])

    else:
        print(glob('../beams/beam_config/GND_{}.npy'.format(gid)))

"""
field = 'S'
cat = v4Scat 
ids = [18169,20960,23102,24148,24622,25053,25884,26914,27965,29928,30144,30152,35046,35579,38472,40985,42548,42985,44471,46275,46500,42113,43114,43683,42607,44133,44725]

for idx in range(len(ids)):
    if len(str(ids[idx])) < 5:
        gid = '0' + str(ids[idx])
    else:
        gid = str(ids[idx])

    W,P,E = Phot_load('G{}D'.format(field), ids[idx], cat)

    mb  = Gen_initial_MB(field, gid)

    Plot_beams(mb, W,P,E, field, gid)
    BMX, Clims, Cspec, Ospec = Beam_cleanup(mb, B_condition=[], R_condition=[])

    np.save('../beams/beam_config/G{}D_{}_ex'.format(field, gid),[BMX])
    np.save('../beams/beam_config/G{}D_{}'.format(field, gid),[Clims, Cspec, Ospec])

    
field = 'N'
cat = v4Ncat 
ids = [12006,14355,17194,20538,21618,22184,22633,23857,26544,34130,35831,37343,38126,38225,14140,33777,37107,12543,15976,26197,38061]

for idx in range(len(ids)):
    if len(str(ids[idx])) < 5:
        gid = '0' + str(ids[idx])
    else:
        gid = str(ids[idx])

    W,P,E = Phot_load('G{}D'.format(field), ids[idx], cat)

    mb  = Gen_initial_MB(field, gid)

    Plot_beams(mb, W,P,E, field, gid)
    BMX, Clims, Cspec, Ospec = Beam_cleanup(mb, B_condition=[], R_condition=[])

    np.save('../beams/beam_config/G{}D_{}_ex'.format(field, gid),[BMX])
    np.save('../beams/beam_config/G{}D_{}'.format(field, gid),[Clims, Cspec, Ospec])"""
