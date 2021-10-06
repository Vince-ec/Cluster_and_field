import fsps
import matplotlib.pyplot as plt
import numpy as np
from spec_exam import F_lam_per_M
from glob import glob
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import seaborn as sea
import pandas as pd
from spec_exam import Gen_spec_2D
from spec_stats import Highest_density_region
from make_sfh_tool import Gen_sim_SFH
import pickle
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def Sig_int(er):
    sig = np.zeros(len(er)-1)
    
    for i in range(len(er)-1):
        sig[i] = np.sqrt(er[i]**2 + er[i+1]**2 )
    
    return np.sum((1/2)*sig)

def SN(w, f, e, wmin, wmax):
    
    IDx = [U for U in range(len(w)) if wmin < w[U] < wmax]
    
    return np.trapz(f[IDx])/ Sig_int(e[IDx])

adb = pd.read_pickle('../dataframes/fitdb/evolution_db.pkl')

Bsn = []
Rsn = []
bsn = 0
rsn = 0

for i in adb.index:
    Gs = Gen_spec_2D(adb.field[i], adb.id[i], adb.zgrism[i], g102_lims=[8200, 11300], g141_lims=[11200, 16000],
            phot_errterm = 0.04, irac_err = 0.08, mask = False)

    if Gs.g102:
        bsn = SN(Gs.Bwv, Gs.Bfl, Gs.Ber,8500,10500)
    if Gs.g141:
        rsn = SN(Gs.Rwv, Gs.Rfl, Gs.Rer,11500,15500)
    
    Bsn.append(bsn)
    Rsn.append(rsn)

    
np.save('../data/all_snr', [Bsn, Rsn])