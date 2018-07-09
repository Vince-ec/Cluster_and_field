#!/home/vestrada78840/miniconda2/envs/astroconda/bin/python 
from C_gen_beams import Gen_DB_and_beams
import numpy as np
import sys

if __name__ == '__main__':
    galaxy = sys.argv[1]
    loc = sys.argv[2]
    ra = sys.argv[3]
    dec = sys.argv[4]

Gen_DB_and_beams( galaxy, loc, ra, dec)