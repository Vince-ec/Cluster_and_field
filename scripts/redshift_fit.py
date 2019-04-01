#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import zfit
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])

zfit(field, galaxy, verbose = True)