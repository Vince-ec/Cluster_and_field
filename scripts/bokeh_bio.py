#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python

import pandas as pd
import numpy as np
from shutil import copy
from glob import glob
from bokeh.models import HoverTool, ColumnDataSource, OpenURL, TapTool
from bokeh import palettes
from bokeh.plotting import figure,save
from bokeh.io import show, output_notebook, output_file
from bokeh.transform import linear_cmap
from bokeh_scripts import RS_img, img_ext
from bokeh.layouts import gridplot
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import astropy.units as u

if __name__ == '__main__':
    IDX = int(sys.argv[1])

alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
morph_db = alldb.query('W_UVJ == "Q" and AGN != "AGN" and lmass >= 10.5 and n_f < 3 and Re < 20 ')

bspec = [27458,294464,36348,48631,19290,32566,32691,33093,26272,35640,45333, 30144]
nog102 = [27714,37189,26139,32799,47223,22774,28890,23073,31452,24033]

inout = []
for i in morph_db.index:     
    if morph_db.id[i] not in bspec and morph_db.id[i] not in nog102: 
        inout.append('i')
    else:
        inout.append('o')
        
morph_db['inout'] = inout
mdb = morph_db.query('inout == "i" and 0.7 < zgrism < 2.5 and Sigma1 > 10**9.6')
import pickle
from spec_tools import Gen_SFH
from spec_exam import Gen_spec
from spec_tools import Posterior_spec

idx = mdb.index[IDX]
rshift = mdb.zgrism[idx]
print('{}-{}'.format(mdb.field[idx], mdb.id[idx]))

with open('../data/SFH/{}_{}_1D.pkl'.format( mdb.field[idx],  mdb.id[idx]), 'rb') as sfh_file:
         SFH = pickle.load(sfh_file)

IMG = RS_img(img_ext(mdb.field[idx], mdb.id[idx]))
zs = [z_at_value(cosmo.lookback_time,(U*u.Gyr + cosmo.lookback_time(mdb.zgrism[idx])), zmax=1E6) for U in SFH.LBT]

src_sfh = ColumnDataSource(data = {'LBT':SFH.LBT,'SFH':SFH.SFH, 'z':zs })

img = figure(plot_width=300, plot_height=350, x_range=(0, 10), y_range=(0, 10), 
                  title = '{}-{}'.format(mdb.field[idx], mdb.id[idx]))
img.image(image=[IMG[::-1]*-1], x=0, y=0, dw=10, dh=10,palette=palettes.gray(100))
img.title.text_font_size = "20pt"
img.xaxis.major_label_text_color = img.yaxis.major_label_text_color = None
img.yaxis.major_tick_line_color = img.xaxis.major_tick_line_color =None 
img.yaxis.minor_tick_line_color = img.xaxis.minor_tick_line_color =None 
#######################
sfh = figure(plot_width = 900, plot_height = 350, x_axis_label ='Lookback Time (Gyr)',
               y_axis_label = 'SFR (M/yr)')
for i in range(100):
    sfh.line(SFH.fulltimes,SFH.sfr_grid[i], color = '#532436', alpha=.075)

r1 = sfh.line(source = src_sfh, x = 'LBT', y='SFH', color = '#C1253C')
sfh.line(SFH.LBT,SFH.SFH_16, color ='black')
sfh.line(SFH.LBT,SFH.SFH_84, color ='black')

sfh.add_tools(HoverTool(tooltips = [('Lookback time', '@LBT'), ('SFH', '@SFH'), ('z', '@z')], renderers = [r1]))
sfh.xaxis.axis_label_text_font_size = "20pt"
sfh.yaxis.axis_label_text_font_size = "20pt"
sfh.xaxis.major_label_text_font_size = "15pt"
sfh.yaxis.major_label_text_font_size = "15pt"
#######################
wave, spec = np.load('../data/allsed/phot/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]))
Pwv, Pflx, Perr = np.load('../data/allsed/phot/{}-{}.npy'.format(mdb.field[idx], mdb.id[idx]))

S1 = figure(plot_width=600, plot_height=350, x_axis_label ='Wavelength (angstrom)',
          y_axis_label = 'F_lambda (10^-18)', tools = "tap,pan,wheel_zoom,box_zoom,reset",x_axis_type="log",
           x_range=(min(Pwv/(1+rshift))*0.95,max(Pwv/(1+rshift))*1.05))

try:
    Bwv, Bfl, Ber = np.load('../data/allsed/g102/{}-{}.npy'.format(mdb.field[idx], mdb.id[idx]))
    Bwv, Bmfl = np.load('../data/allsed/g102/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]))
    S1.circle(Bwv/(1+rshift), Bfl *1E18,color='#36787A',size = 3, alpha = 1, line_width = 1.5, line_color = 'black')
    S1.segment(Bwv/(1+rshift), (Bfl-Ber)*1E18, Bwv/(1+rshift),(Bfl+Ber)*1E18,color='#36787A')
    S1.line(Bwv/(1+rshift), Bmfl*1E18, color ='black', line_width = 2, alpha = 0.8)
    IDB = [U for U in range(len(wave)) if wave[U] < Bwv[0]/(1+rshift)]
except:
    IDB = [U for U in range(len(wave)) if wave[U] < Rwv[0]/(1+rshift)]
    
try:
    Rwv, Rfl, Rer = np.load('../data/allsed/g141/{}-{}.npy'.format(mdb.field[idx], mdb.id[idx]))
    Rwv, Rmfl = np.load('../data/allsed/g141/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]))
    S1.circle(Rwv/(1+rshift),Rfl*1E18,color='#EA2E3B',size = 3, alpha = 1, line_width = 1.5, line_color = 'black')
    S1.segment(Rwv/(1+rshift),(Rfl-Rer)*1E18,Rwv/(1+rshift),(Rfl+Rer)*1E18,color='#EA2E3B')
    S1.line(Rwv/(1+rshift), Rmfl *1E18, color ='black', line_width = 2, alpha = 0.8)
    IDR = [U for U in range(len(wave)) if wave[U] > Rwv[-1]/(1+rshift)]
except:
    IDR = [U for U in range(len(wave)) if wave[U] > Bwv[-1]/(1+rshift)]
    
S1.circle(Pwv/(1+rshift),Pflx *1E18,color='#685877', size = 10, alpha = 1, line_width = 1.5, line_color = 'black')
S1.segment(Pwv/(1+rshift),(Pflx-Perr)*1E18,Pwv/(1+rshift),(Pflx+Perr)*1E18,color='#685877')
S1.line(wave[IDB], spec[IDB]*1E18, color ='black', line_width = 2, alpha = 0.8)
S1.line(wave[IDR], spec[IDR]*1E18, color ='black', line_width = 2, alpha = 0.8)
S1.xaxis.ticker = [2500,5000,10000,25000]
S1.yaxis.axis_label_text_font_size=S1.xaxis.axis_label_text_font_size="20pt"
S1.yaxis.major_label_text_font_size=S1.xaxis.major_label_text_font_size="15pt"

output_file('../../Vince-ec.github.io/appendix/sfhs/{}.html'.format(mdb.id[idx]),
                title = '{}-{}'.format(mdb.field[idx], mdb.id[idx]))
save(gridplot([[img, S1],[sfh]]))