import pandas as pd
import numpy as np
from spec_stats import LOWESS_trend

data_path = '/scratch/user/vestrada78840/data/'
out_path = '/scratch/user/vestrada78840/trends/'

fdb = pd.read_pickle(data_path + 'evolution_db.pkl')
fdb = fdb.query('AGN != "AGN" and lmass > 10 and concen < -0.4 and 0.7 < zgrism < 2.7')

########mass metal quiescent#################################################################
##z0.7-1
X=fdb.query('sf_prob_2 < 0.2 and 0.7 < zgrism < 1.0').lmass.values
Y=np.log10(fdb.query('sf_prob_2 < 0.2 and 0.7 < zgrism < 1.0').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_mass_metal_Q_z0.7-1.npy', smooth_grid, allow_pickle = True)

##z1-1.5
X=fdb.query('sf_prob_2 < 0.2 and 1.0 < zgrism < 1.5').lmass.values
Y=np.log10(fdb.query('sf_prob_2 < 0.2 and 1.0 < zgrism < 1.5').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_mass_metal_Q_z1-1.5.npy', smooth_grid, allow_pickle = True)

##z1.5-2
X=fdb.query('sf_prob_2 < 0.2 and 1.5 < zgrism < 2.0').lmass.values
Y=np.log10(fdb.query('sf_prob_2 < 0.2 and 1.5 < zgrism < 2.0').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_mass_metal_Q_z1.5-2.npy', smooth_grid, allow_pickle = True)

##z2-2.6
X=fdb.query('sf_prob_2 < 0.2 and 2 < zgrism < 2.6').lmass.values
Y=np.log10(fdb.query('sf_prob_2 < 0.2 and 2 < zgrism < 2.6').Z.values)

Sx_range = np.linspace(min(X), max(X), 1000)
smooth_grid = []
for i in range(1000):
    IDX = np.random.choice(np.arange(len(X)),len(X))
    x, y = LOWESS(X[IDX], Y[IDX], alpha = 0.6666)
    smooth_grid.append(interp1d(x, y, bounds_error=False, fill_value=0)(Sx_range))
    
np.save(out_path + 'TREND_mass_metal_Q_z2-2.6.npy', smooth_grid, allow_pickle = True)