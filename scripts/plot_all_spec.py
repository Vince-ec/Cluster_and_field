import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from glob import glob 
from spec_extract import Stack
from astropy.io import fits

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

def Extract_spec(Field, galaxy_id):
    spec_list = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/*{}*/*{}*.1D.fits'.format(Field, galaxy_id))

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

    if len(Bwv) > 0 and len(Rwv) == 0:                
        SBW, SBF, SBE, SBT, SBL, SBC = Stack(Bwv, Bfl, Ber, Bft, Bln, Bct)
        IDX = [U for U in range(len(SBW)) if 8500 < SBW[U] < 11100 and SBF[U] > 0 and SBF[U]/SBE[U] > .1]
        return SBW[IDX], SBF[IDX] / SBT[IDX], SBE[IDX] / SBT[IDX]

    if len(Rwv) > 0 and len(Bwv) == 0:     
        SRW, SRF, SRE, SRT, SRL, SRC = Stack(Rwv, Rfl, Rer, Rft, Rln, Rct)
        IDX = [U for U in range(len(SRW)) if 11100 < SRW[U] < 16500 and SRF[U] > 0 and SRF[U]/SRE[U] > .1]
        return SRW[IDX], SRF[IDX] / SRT[IDX], SRE[IDX] / SRT[IDX]
    
    if len(Rwv) > 0 and len(Bwv) > 0:     
        
        SBW, SBF, SBE, SBT, SBL, SBC = Stack(Bwv, Bfl, Ber, Bft, Bln, Bct)
        SRW, SRF, SRE, SRT, SRL, SRC = Stack(Rwv, Rfl, Rer, Rft, Rln, Rct)
        
        IDB = [U for U in range(len(SBW)) if 8500 < SBW[U] < 11100 and SBF[U] > 0 and SBF[U]/SBE[U] > .1]
        IDR = [U for U in range(len(SRW)) if 11100 < SRW[U] < 16500 and SRF[U] > 0 and SRF[U]/SRE[U] > .1]
        
        return [SBW[IDB],SRW[IDR]], [SBF[IDB] / SBT[IDB],SRF[IDR] / SRT[IDR]], [SBE[IDB] / SBT[IDB],SRE[IDR] / SRT[IDR]]

        
v4Ncat = Table.read('/Volumes/Vince_CLEAR/3dhst_V4.4/goodsn_3dhst.v4.4.cats/Catalog/goodsn_3dhst.v4.4.cat',
                 format='ascii').to_pandas()
v4Scat = Table.read('/Volumes/Vince_CLEAR/3dhst_V4.4/goodss_3dhst.v4.4.cats/Catalog/goodss_3dhst.v4.4.cat',
                 format='ascii').to_pandas()

GND_all = pd.read_pickle('../dataframes/galaxy_frames/GND_CLEAR.pkl')
GSD_all = pd.read_pickle('../dataframes/galaxy_frames/GSD_CLEAR.pkl')

for i in GSD_all.index[:10]:
    if len(str(GSD_all.id[i])) < 5:
        gid = '0' + str(GSD_all.id[i])
    else:
        gid = str(GSD_all.id[i])
    
    GWV,GFL,GER = Extract_spec('S', gid)
    
    W,P,E = Phot_load('GSD', GSD_all.id[i], v4Scat)

    plt.figure(figsize=[12,8])
    plt.errorbar(W,P,E, fmt = 'o',ms=10)
    
    for ii in range(len(GWV)):
        plt.errorbar(GWV[ii],GFL[ii],GER[ii], fmt = 'o',ms = 1, color = 'k')
    
    plt.xscale('log')
    plt.title('GSD-' + str(GSD_all.id[i],), fontsize = 20)
    plt.xticks([5000,10000,25000,50000],[5000,10000,25000,50000])
    plt.savefig('../plots/allspec_exam/GSD-{}'.format(GSD_all.id[i]), bbox_inches = 'tight')
"""
for i in GND_all.index:
    W,P,E = Phot_load('GND', GND_all.id[i], v4Ncat)

    plt.figure(figsize=[12,8])
    plt.errorbar(W,P,E, fmt = 'o',ms=10)
    plt.xscale('log')
    plt.title('GND-' + str(GND_all.id[i],), fontsize = 20)
    plt.xticks([5000,10000,25000,50000],[5000,10000,25000,50000])
    plt.savefig('../plots/allspec_exam/GND-{}'.format(GND_all.id[i]), bbox_inches = 'tight')"""