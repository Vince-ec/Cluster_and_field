import pandas as pd
import numpy as np
from spec_stats import LOWESS_trend

data_path = '/scratch/user/vestrada78840/data/'
out_path = '/scratch/user/vestrada78840/trends/'

fdb = pd.read_pickle(data_path + 'evolution_db.pkl')
fdb = fdb.query('AGN != "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')

####### concentration metallicity#################################################################
##HM
X=fdb.query('lmass > 11').concen.values
Y=np.log10(fdb.query('lmass > 11').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_concen_metal_HM.npy', smooth_grid, allow_pickle = True)

##MM
X=fdb.query('10.5 < lmass < 11').concen.values
Y=np.log10(fdb.query('10.5 < lmass < 11').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_concen_metal_MM.npy', smooth_grid, allow_pickle = True)

##LM
X=fdb.query('10.0 < lmass < 10.5').concen.values
Y=np.log10(fdb.query('10.0 < lmass < 10.5').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_concen_metal_LM.npy', smooth_grid, allow_pickle = True)