#!/home/vestrada78840/miniconda2/envs/astroconda/bin/python 
import numpy as np
import sys
from C_z_fit_beams import Z_fit

metal=np.array([0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
age=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
z = np.arange(1.0,2.005,0.005)

if __name__ == '__main__':
    galaxy = sys.argv[1]
    loc = sys.argv[2]
    
Z_fit(galaxy, loc, z, metal, age)