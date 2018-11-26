#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from C_full_fit import Fit_rshift
import os
import numpy as np
import sys
from glob import glob
    
### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    beam_path = '/fdata/scratch/vestrada78840/clear_q_beams/'

else:
    beam_path = '../beams/'

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])

z = np.arange(1.3,3.5,0.01)
ttest=[0, 8.95, 9.26, 9.43]
metal=np.round(np.arange(0.002,0.031,0.001),3)
age=np.round(np.arange(.5,4.6,.1),1)

Fit_rshift(field, galaxy, glob(beam_path + '*{0}*.g102.A.fits'.format(galaxy))[0],
              glob(beam_path + '*{0}*.g141.A.fits'.format(galaxy))[0], metal, age, ttest, z)
    