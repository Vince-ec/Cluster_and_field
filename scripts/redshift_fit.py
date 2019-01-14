#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import Redshift_fitter
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    beam_path = '/fdata/scratch/vestrada78840/beams/'

else:
    beam_path = '../beams/'

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])

try:
    g102_beam = glob( beam_path + '*{0}*g102*'.format(galaxy))[0]
except:
    g102_beam = ''

try:
    g141_beam = glob( beam_path + '*{0}*g141*'.format(galaxy))[0]
except:
    g141_beam = ''

Redshift_fitter(field, galaxy, g102_beam, g141_beam, errterm=0.03, decontam = True)
    
