#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import Best_fitter_sim
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'

if hpath == '/home/vestrada78840/':
    beam_path = '/fdata/scratch/vestrada78840/beams/'
    spec_path = '/fdata/scratch/vestrada78840/stack_specs/'

else:
    beam_path = '../beams/'
    spec_path = '../spec_files/'
    
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    simZ = float(sys.argv[4])
    simt = float(sys.argv[5])
    simtau = float(sys.argv[6])
    simz = float(sys.argv[7])
    simd  = float(sys.argv[8])

Best_fitter_sim(field, galaxy, glob( beam_path + '*{0}*g102*'.format(galaxy))[0], 
                glob( beam_path + '*{0}*g141*'.format(galaxy))[0], 
                specz, simZ, simt, simtau, simz, simd,
                mod = 0, errterm=0.03, decontam = False)
