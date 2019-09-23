import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from glob import glob
import seaborn as sea
import os
sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True)


### set home for files
hpath = os.environ['HOME'] + '/'

from spec_extract import Stack

def Extract_g102(field, galaxy):
    if field[1] == 'N':
        pre = 'N'
    else:
        pre = 'S'
    spec_list = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/*{0}*/*{1}*.1D.fits'.format(pre, galaxy))

    Bwv, Bfl, Ber, Bft, Bln, Bct = [[],[],[],[],[],[]]

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

    if len(Bwv) > 0:                
        return Stack(Bwv, Bfl, Ber, Bft, Bln, Bct)

def Extract_g141(field, galaxy):
    if field[1] == 'N':
        pre = 'N'
    else:
        pre = 'S'
    spec_list = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/*{0}*/*{1}*.1D.fits'.format(pre, galaxy))

    Rwv, Rfl, Rer, Rft, Rln, Rct = [[],[],[],[],[],[]]

    for i in range(len(spec_list)):
        dat = fits.open(spec_list[i])

        try:
            Rwv.append(np.array(dat['G141'].data['wave']).T)
            Rfl.append(np.array(dat['G141'].data['flux']).T)
            Rer.append(np.array(dat['G141'].data['err']).T)
            Rft.append(np.array(dat['G141'].data['flat']).T)
            Rln.append(np.array(dat['G141'].data['line']).T)
            Rct.append(np.array(dat['G141'].data['cont']).T)

        except:
            print('no g141')

    if len(Rwv) > 0:     
        return Stack(Rwv, Rfl, Rer, Rft, Rln, Rct)

# SBW, SBF, SBE, SBT, SBL, SBC



GS_buldge= pd.read_pickle('../dataframes/galaxy_frames/GS_buldge.pkl')
GN_buldge= pd.read_pickle('../dataframes/galaxy_frames/GN_buldge.pkl')

mdb = pd.read_pickle('../dataframes/fitdb/fullfitdb.pkl')
SFdb = pd.read_pickle('../Casey_data/SF_db_p1.pkl')


for i in GS_buldge.index:
    if GS_buldge.id[i] not in mdb.query('field == "GSD"').id.values and GS_buldge.id[i] not in SFdb.query('field == "GSD"').id.values:
        G102 = Extract_g102('GSD', GS_buldge.id[i])
        G141 = Extract_g141('GSD', GS_buldge.id[i])      
        
        photz = GS_buldge.zphot[i]
        fig, ax = plt.subplots(figsize = [12,6])
        if G102 is not None:
            ax.errorbar(G102[0] / (1+photz), G102[1] / G102[3], G102[2] / G102[3], linestyle = 'none', color = 'b')
        
        if G141 is not None:
            ax.errorbar(G141[0] / (1+photz), G141[1] / G141[3], G141[2] / G141[3], linestyle = 'none', color = 'r')
        
        ax.axvline(3727.092 ,linestyle='--', alpha=.3)
        ax.axvline(3867 ,linestyle='--', alpha=.3)
        ax.axvline(4102.89 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(4341.68 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(4862.68 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(5008.240 ,linestyle='--', alpha=.3)
        ax.axvline(6564.61 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(6718.29,linestyle='--', alpha=.3, color = 'k')
        fig.savefig('../plots/buldge_spec/GSD_{}.png'.format(GS_buldge.id[i]))
     
        
for i in GN_buldge.index:
    if GN_buldge.id[i] not in mdb.query('field == "GND"').id.values and GN_buldge.id[i] not in SFdb.query('field == "GND"').id.values:
        G102 = Extract_g102('GND', GN_buldge.id[i])
        G141 = Extract_g141('GND', GN_buldge.id[i])      
        
        photz = GN_buldge.zphot[i]
        fig, ax = plt.subplots(figsize = [12,6])
        if G102 is not None:
            ax.errorbar(G102[0] / (1+photz), G102[1] / G102[3], G102[2] / G102[3], linestyle = 'none', color = 'b')
        
        if G141 is not None:
            ax.errorbar(G141[0] / (1+photz), G141[1] / G141[3], G141[2] / G141[3], linestyle = 'none', color = 'r')
        
        ax.axvline(3727.092 ,linestyle='--', alpha=.3)
        ax.axvline(3867 ,linestyle='--', alpha=.3)
        ax.axvline(4102.89 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(4341.68 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(4862.68 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(5008.240 ,linestyle='--', alpha=.3)
        ax.axvline(6564.61 ,linestyle='--', alpha=.3, color = 'r')
        ax.axvline(6718.29,linestyle='--', alpha=.3, color = 'k')
        fig.savefig('../plots/buldge_spec/GND_{}.png'.format(GN_buldge.id[i]))