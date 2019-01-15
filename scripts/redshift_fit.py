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

g102_beam = glob( beam_path + '*{0}*g102*'.format(galaxy))
g141_beam = glob( beam_path + '*{0}*g141*'.format(galaxy))

idx = 0
for g1 in g102_beam:
    for g2 in g141_beam:
        Redshift_fitter(field, galaxy, g1, g2, mod = idx, errterm=0.03, decontam = True)
        idx+=1
    
