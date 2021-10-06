import pandas as pd
import numpy as np
from spec_stats import LOWESS_trend

data_path = '/scratch/user/vestrada78840/data/'
out_path = '/scratch/user/vestrada78840/trends/'

fdb = pd.read_pickle(data_path + 'evolution_db.pkl')
fdb = fdb.query('AGN != "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')

########morphology formation#################################################################
##z50_concen
X=fdb.concen.values
Y=fdb.z50.values

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_z50_concen.npy', smooth_grid, allow_pickle = True)

##z50_mass
X=fdb.lmass.values
Y=fdb.z50.values

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_z50_mass.npy', smooth_grid, allow_pickle = True)

##z50_sigma1
X=fdb.log_Sigma1.values
Y=fdb.z50.values

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_z50_mass.npy', smooth_grid, allow_pickle = True)