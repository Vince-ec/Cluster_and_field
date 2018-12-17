#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
import numpy as np
from C_sim_test import Fit_all_sim, Analyze_grism_fit, Analyze_indv_chi
from spec_tools import Oldest_galaxy
import os
import sys
from glob import glob
from time import time
    
### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    data_path = '/fdata/scratch/vestrada78840/data/'

else:
    beam_path = '../beams/'
    data_path = '../data/'

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    rshift = float(sys.argv[3])
    simZ = float(sys.argv[4])
    simt = float(sys.argv[5])
    simtau = float(sys.argv[6])
    simz = float(sys.argv[7])
    simd = float(sys.argv[8])

tau = np.round(np.logspace(-2,np.log10(3),20),3)
age = np.round(np.arange(0.1, np.round(Oldest_galaxy(rshift),1) + .05,.05),2)
metal=np.round(np.arange(0.002 , 0.0305, 0.0005),4)
dust = np.arange(0, 1.05, 0.05)

Fit_all_sim(field, galaxy, glob(beam_path + '*{0}*.g102.A.fits'.format(galaxy))[0],
              glob(beam_path + '*{0}*.g141.A.fits'.format(galaxy))[0], rshift, 
            metal, age, tau, dust, simZ, simt, simtau, simz, simd, 
            'fit_test_{0}_{1}'.format(field, galaxy), gen_models = False, 
             errterm = 0.03, outname = 'sim_fit_{0}_{1}'.format(field, galaxy))

Analyze_grism_fit('sim_fit_{0}_{1}'.format(field, galaxy), metal, age, tau, rshift, dust,
                  age_conv=data_path + 'fit_test_{0}_{1}_lwa_scale.npy'.format(field, galaxy))
Analyze_indv_chi('sim_fit_{0}_{1}'.format(field, galaxy), metal, age, tau, rshift, dust, instr = 'g102',
                age_conv=data_path + 'fit_test_{0}_{1}_lwa_scale.npy'.format(field, galaxy))  
Analyze_indv_chi('sim_fit_{0}_{1}'.format(field, galaxy), metal, age, tau, rshift, dust, instr = 'g141',
                age_conv=data_path + 'fit_test_{0}_{1}_lwa_scale.npy'.format(field, galaxy))
Analyze_indv_chi('sim_fit_{0}_{1}'.format(field, galaxy), metal, age, tau, rshift, dust, instr = 'phot',
                age_conv=data_path + 'fit_test_{0}_{1}_lwa_scale.npy'.format(field, galaxy))