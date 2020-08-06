import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import astropy.units as u
from scipy.interpolate import interp1d, interp2d
from glob import glob
import os
from bokeh.models import ColumnDataSource, Select
from bokeh import palettes
from bokeh.plotting import figure,save
from bokeh.io import show, output_notebook, output_file, curdoc
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot, column, row

alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
sfdb = pd.read_pickle('../dataframes/fitdb/SFfits_1D.pkl')

inout = []
for i in alldb.index:
    IO = 'i'
    if alldb.field[i] == 'GND' and alldb.id[i] in sfdb.query('field == "GND"').id.values:
        IO = 'o' 
    if alldb.field[i] == 'GSD' and alldb.id[i] in sfdb.query('field == "GSD"').id.values:
        IO = 'o' 
    inout.append(IO)
    
alldb['inout'] = inout

Qdb = alldb.query('inout == "i" and t_50 > 0')

adb = pd.concat([Qdb, sfdb])
adb = adb.reset_index()
adb = adb.drop(columns='index')

for i in adb.index:
    if adb.zgrism[i] < 0:
        adb.zgrism[i] = adb.zfit[i]

qdb = adb.query('log_ssfr < -11 and AGN != "AGN"')
sfdb = adb.query('log_ssfr > -11 and AGN != "AGN"')
xqdb = adb.query('log_ssfr < -11 and AGN == "AGN"')
xsdb = adb.query('log_ssfr > -11 and AGN == "AGN"')

source = ColumnDataSource(data={'x': qdb['zgrism'], 'y': qdb['t_50']})
# Create plots and widgets
plot = figure()
plot.circle(x='x', y='y', source=source)
menu = Select(options=['t_50','lmass', 'z_50', 'lwa'],
 value='t_50', title='Y-value')


# Add callback to widgets
def callback(attr, old, new):
    print(menu.value)
    source.data={'y': qdb[menu.value]}

menu.on_change('value', callback)
# Arrange plots and widgets in layouts
layout = column(menu, plot)
curdoc().add_root(layout)
