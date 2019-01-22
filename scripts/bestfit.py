#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import Best_fitter, Get_bestfits
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

if os.path.isfile(spec_path + '{0}_{1}_g102.npy'.format(field,galaxy)) and len(np.load(spec_path + '{0}_{1}_g102.npy'.format(field,galaxy))) > 5:
    g102_beam = glob( beam_path + '*{0}*g102*'.format(galaxy))
else:
    g102_beam = []
    
    
if os.path.isfile(spec_path + '{0}_{1}_g141.npy'.format(field,galaxy)) and len(np.load(spec_path + '{0}_{1}_g141.npy'.format(field,galaxy))) > 5:
    g141_beam = glob( beam_path + '*{0}*g141*'.format(galaxy))
else:
    g141_beam = []
    
if len(g102_beam) < 1:
    idx = 0
    for g2 in g141_beam:
        Best_fitter(field, galaxy, '', g2, specz, mod = idx, errterm=0.03, decontam = True)
        idx+=1

if len(g141_beam) < 1:
    idx = 0
    for g1 in g102_beam:
        Best_fitter(field, galaxy, g1, '', specz, mod = idx, errterm=0.03, decontam = True)
        idx+=1
    
if len(g102_beam) >= 1 and len(g141_beam) >= 1:   
    idx = 0
    for g1 in g102_beam:
        for g2 in g141_beam:
            Best_fitter(field, galaxy, g1, g2, specz, mod = idx, errterm=0.03, decontam = True)
            idx+=1
            
Get_bestfits(field, galaxy)