import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from glob import glob

adb = pd.read_pickle('../dataframes/fitdb/evolution_db.pkl')
# Adb = adb.query('AGN != "AGN" and lmass > 10 and 0.7 < zgrism < 2.7')
Adb = adb.query('AGN != "AGN" and lmass > 10 and 0.7 < zgrism < 1.5 and id != 30081')

for i in Adb.index:
    line_list = glob('/Volumes/Vince_CLEAR/RELEASE_v3.0.0/1Dspec/*{}*/*{}*'.format(Adb.field[i][1], Adb.id[i]))
    print(line_list,Adb.field[i][1], Adb.id[i])
    z = Adb.zgrism[i]
    loc_line ='----'
    Ha_loc = 6564 * (1+z)

    if 8000 < Ha_loc < 11300:
        loc_line = 'G102'
    if 11300 < Ha_loc < 16000:
        loc_line = 'G141'

    if loc_line == '----':
        pass

    else:
        for ii in range(len(line_list)):
            oned = fits.open(line_list[ii])
            try:
                dat = oned[loc_line].data
                wv = dat['wave']
                ln = dat['line']
                ct = dat['cont']
                ft = dat['flat']

                wv = wv[ln > 0]
                ct = ct[ln > 0]
                ft = ft[ln > 0]
                ln = ln[ln > 0]
                plt.figure()
                plt.plot(wv / (1+z),ln/ft-ct/ft)
                plt.axvline(6100)
                plt.axvline(6900)
                plt.savefig('../plots/Ha_plots/{}_{}.png'.format(Adb.field[i], Adb.id[i]), bbox_inches = 'tight')    
            except:
                print('no line')