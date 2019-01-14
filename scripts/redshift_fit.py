from spec_id import Redshift_fitter
import numpy as np
from glob import glob
import pandas as pd
import os

gsd_cat = pd.read_pickle('../dataframes/GS_snr.pkl')
gnd_cat = pd.read_pickle('../dataframes/GN_snr.pkl')

for i in gsd_cat.index:

    try:
        g102_beam = glob('../beams/*{0}*g102*'.format(gsd_cat.id[i]))[0]
    except:
        g102_beam = ''

    try:
        g141_beam = glob('../beams/*{0}*g141*'.format(gsd_cat.id[i]))[0]
    except:
        g141_beam = ''

    Redshift_fitter('GSD',gsd_cat.id[i],g102_beam, g141_beam, errterm=0.03, decontam = False)
    
for i in gnd_cat.index:

    try:
        g102_beam = glob('../beams/*{0}*g102*'.format(gnd_cat.id[i]))[0]
    except:
        g102_beam = ''

    try:
        g141_beam = glob('../beams/*{0}*g141*'.format(gnd_cat.id[i]))[0]
    except:
        g141_beam = ''

    Redshift_fitter('GND',gnd_cat.id[i],g102_beam, g141_beam, errterm=0.03, decontam = False)