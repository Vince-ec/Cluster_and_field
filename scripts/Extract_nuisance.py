#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import fsps
import numpy as np
from glob import glob
import pandas as pd
import os
import sys
from grizli import multifit
from grizli.utils import SpectrumTemplate

#################################
#########BALMER fit slope########
#################################
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    lines = sys.argv[4:]

###########gen spec##########
Gs = Gen_spec_2D(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 
    
MBS = Gather_MB_data(Gs)      
#############multifit###############
wave0 = 4000

if len(lines) > 0:
    Q_temps = Gen_temp_dict_addline(lines = lines)
else:
    Q_temps = {}

dres = np.load( out_path + '{0}_{1}_tabMfit.npy'.format(field, galaxy)).item()

#### gen light-weighted age posterior
if not os.path.isfile(pos_path + '{0}_{1}_tabMfit_Plwa.npy'.format(field, galaxy)):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

    lwa = []
    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr = dres.samples[i]
        lwa.append(get_lwa([m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], get_agebins(a),sp)[0])

    t,pt = Get_derived_posterior(np.array(lwa), dres)
    np.save(pos_path + '{0}_{1}_tabMfit_Plwa'.format(field, galaxy),[t,pt])

#### gen logmass posterior
if not os.path.isfile(pos_path + '{0}_{1}_tabMfit_Plm.npy'.format(field, galaxy)):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    lm = []
    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, sb, sr = dres.samples[i]

        sp.params['dust2'] = d
        sp.params['dust1'] = d
        sp.params['logzsol'] = np.log10(m)

        time, sfr, tmax = convert_sfh(get_agebins(a), [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10], maxage = a*1E9)

        sp.set_tabular_sfh(time,sfr) 

        wave, flux = sp.get_spectrum(tage = a, peraa = True)

        flam = F_lam_per_M(flux, wave * (1+z), z, 0, sp.stellar_mass)
        Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flam)
        scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
        lm.append(np.log10(scl))

    t,pt = Get_derived_posterior(np.array(lm), dres)
    np.save(pos_path + '{0}_{1}_tabMfit_Plm'.format(field, galaxy),[t,pt])

#### gen line posteriors
L = lines
print(glob(pos_path + '{0}_{1}_tabMfit_Pline*'.format(field, galaxy)))
if len(L) > 0 and len(glob(pos_path + '{0}_{1}_tabMfit_Pline*'.format(field, galaxy))) == 0:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

    lines = []
    for i in range(len(MBS)):
        lines.append({})    
        for k in Q_temps:
            if k[0] == 'l':
                lines[i][k] = []

    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, z, d, s1, s2 = dres.samples[i]
        wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10])

        Gchi2, fit = Fit_MB(MBS, [s1, s2], Q_temps, wave, flux, z)

        for ii in range(len(lines)):
            for k in lines[ii]:
                lines[ii][k].append(fit[ii]['cfit'][k][0])

    for i in range(len(lines)):
        for k in lines[i]:
            if sum(lines[i][k]) > 0:
                x,px = Get_derived_posterior(np.array(lines[i][k]), dres)
                np.save(pos_path + '{}_{}_tabMfit_P{}_{}'.format(field, galaxy, k, i),[x,px])
