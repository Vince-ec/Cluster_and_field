import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sea
import os
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})

from matplotlib.colors import ListedColormap
clist = [[166, 58, 0],
[72, 146, 86],
[20, 57, 80]]
X = np.linspace(-15, -7, 500)

A = []
B = []
C = []
for i in range(2):
    A.extend(np.linspace(clist[i][0]/255, clist[i+1][0]/255, 500))
    B.extend(np.linspace(clist[i][1]/255, clist[i+1][1]/255, 500))
    C.extend(np.linspace(clist[i][2]/255, clist[i+1][2]/255, 500))
CMAP = ListedColormap(np.array([A,B,C]).T)

Adb = pd.read_pickle('../dataframes/fitdb/evolution_db_masslim.pkl')
Adb = Adb.query('id != 44707')
zdb = Adb.sort_values('zgrism')

Zdx = zdb.index[0 :150]

kde =stats.gaussian_kde(Adb.log_ssfr[Zdx])

y = kde(X)
Y0 = np.array(y)
for i in range(len(Y0)):
    if i < 250:
        Y0[i] = 0
mu = X[Y0 == max(Y0)]

idx = 0

while len(zdb.index[0 + idx:150 + idx]) == 150:
    Zdx = zdb.index[0 + idx:150 + idx]
    
    
    plt.figure(figsize = [16,9])
    ax1=plt.subplot()
    sea.distplot(Adb.log_ssfr[Zdx], ax = ax1, kde_kws = {'linewidth': 5}, bins = 20, color = CMAP(idx/304),
                 label = '{:1.2f} '.format(min(Adb.zgrism[Zdx])) + '< z$_{grism}$ < ' + '{:1.2f}'.format(max(Adb.zgrism[Zdx]) ))

    ax1.set_xlabel('log(sSFR (yr$^{-1}$))',size=25)
    ax1.set_ylabel('P(log(sSFR (yr$^{-1}$)))',size=25)
    ax1.tick_params(axis='both', which='major', labelsize=17)

    ax1.set_xlim(-13.5, -8)
    ax1.set_ylim(0, 0.9) 
    
    ax1.legend(fontsize = 18, loc = 2)

    ax1.axvline(mu, color = 'k', alpha = 0.5, linestyle = '--', linewidth = 5)
    
    kde =stats.gaussian_kde(Adb.log_ssfr[Zdx])

    y = kde(X)
    Y0 = np.array(y)
    for i in range(len(Y0)):
        if i < 250:
            Y0[i] = 0
    mu2 = X[Y0 == max(Y0)]
    ax1.axvline(mu2, color = CMAP(idx/304), alpha = 0.5, linewidth = 5)
    ax1.fill_between([mu[0], mu2[0]], [0.9,0.9],color = 'k', alpha = 0.3)
    
    plt.savefig('../plots/evolution_plots/ssfr_gif/gif_sSFR_{}.png'.format(idx), bbox_inches = 'tight')   
    
    
    idx +=1
    
#     if idx == 5:
#         break