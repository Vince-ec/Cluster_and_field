import numpy as np
from glob import glob
import pandas as pd

slist = glob('fullfit_*.slrm')

zfit = []
field = []
galaxy = []

for fl in slist:
    f = open(fl, "r")

    idx = 0
    for i in f:
        if idx == 11:
            line = i
        idx+=1

    zfit.append(float(line.split(' ')[-1]))
    field.append(line.split(' ')[-3])    
    galaxy.append(int(line.split(' ')[-2]))    
    
DB = pd.DataFrame({'field' : field, 'id' : galaxy, 'zfit' : zfit}) 

print(DB)

sfh_path = '/fdata/scratch/vestrada78840/SFH/'

DB.to_pickle(sfh_path + 'zfit_catalog.pkl')