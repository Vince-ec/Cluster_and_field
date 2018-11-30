#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
import numpy as np
from C_full_fit import Fit_all2
import os
import sys
from glob import glob
    
### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    beam_path = '/fdata/scratch/vestrada78840/beams/'

else:
    beam_path = '../beams/'

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    rshift = float(sys.argv[3])

#z = np.arange(rshift - 0.04, rshift + 0.041, 0.001)
age=np.round(np.arange(.5,6.1,.1),1)
tau=[0,8.0, 8.3, 8.48, 8.6, 8.7, 8.78, 8.85, 8.9, 8.95, 9.0, 9.04, 9.08, 9.11, 9.15, 9.18, 9.2, 9.23, 9.26, 9.28,
     9.3, 9.32, 9.34, 9.36, 9.38, 9.4, 9.41, 9.43, 9.45, 9.46, 9.48]
dust = np.arange(0, 1.1, 0.1)
#metal=np.round(np.arange(0.002,0.031,0.001),3)

metal =np.round(np.arange(0.002,0.031,0.008),3)
z = [1.6,1.606,1.61]

Fit_all2(field, galaxy, glob(beam_path + '*{0}*.g102.A.fits'.format(galaxy))[0],
              glob(beam_path + '*{0}*.g141.A.fits'.format(galaxy))[0], rshift, 
        metal, age, tau, z, dust, 'fit_test_{0}_{1}'.format(field, galaxy),
         outname = 'fit_test2_{0}_{1}'.format(field, galaxy) , gen_models = False)