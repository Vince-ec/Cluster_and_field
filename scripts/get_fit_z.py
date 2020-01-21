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
########################
slist = glob('B_Q/bq_fit_*.slrm')

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
########################
slist = glob('B_SF/bsf_fit_*.slrm')

for fl in slist:
    f = open(fl, "r")

    idx = 0
    for i in f:
        if idx == 11:
            line = i
        idx+=1

    zfit.append(float(line.split(' ')[-2]))
    field.append(line.split(' ')[-4])    
    galaxy.append(int(line.split(' ')[-3]))    
    
########################

slist = glob('SFfit_*.slrm')

for fl in slist:
    if fl.split('_')[1] != 'sim':
        f = open(fl, "r")

        idx = 0
        for i in f:
            if idx == 11:
                line = i
            idx+=1

        zfit.append(float(line.split(' ')[-2]))
        field.append(line.split(' ')[-4])    
        galaxy.append(int(line.split(' ')[-3]))  
    
    
DB = pd.DataFrame({'field' : field, 'id' : galaxy, 'zfit' : zfit}) 

print(DB)

sfh_path = '/fdata/scratch/vestrada78840/SFH/'

DB.to_pickle(sfh_path + 'zfit_catalog.pkl')