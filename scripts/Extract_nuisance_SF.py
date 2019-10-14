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
hpath = os.environ['HOME'] + '/'
  
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])
    logmass = float(sys.argv[4])
    
verbose=False
poolsize = 8

#############
###########gen spec##########
Gs = Gen_spec_2D(field, galaxy, 1, phot_errterm = 0.04, irac_err = 0.08) 
    
MBS = Gather_MB_data(Gs)   

########
wave0 = 4000
SF_temps =  Gen_temp_dict(specz, 8000, 16000)

dres = np.load(out_path + '{0}_{1}_SFMfit.npy'.format(field, galaxy)).item() 

#### gen light-weighted age posterior

if not os.path.isfile(pos_path + '{0}_{1}_SFMfit_Plwa.npy'.format(field, galaxy)):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    lwa = []
    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]
        lwa.append(get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum = 6),sp)[0])

    t,pt = Get_derived_posterior(np.array(lwa), dres)
    np.save(pos_path + '{0}_{1}_SFMfit_Plwa'.format(field, galaxy),[t,pt])

#### gen logmass posterior
if not os.path.isfile(pos_path + '{0}_{1}_SFMfit_Plm.npy'.format(field, galaxy)):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    lm = []
    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]

        sp.params['dust2'] = d
        sp.params['logzsol'] = np.log10(m)

        time, sfr, tmax = convert_sfh(get_agebins(a, binnum = 6), [m1, m2, m3, m4, m5, m6], maxage = a*1E9)

        sp.set_tabular_sfh(time,sfr) 

        wave, flux = sp.get_spectrum(tage = a, peraa = True)

        flam = F_lam_per_M(flux, wave * (1+z), z, 0, sp.stellar_mass)
        Pmfl = Gs.Sim_phot_mult(wave * (1 + z),flam)
        scl = Scale_model(Gs.Pflx, Gs.Perr, Pmfl)
        lm.append(np.log10(scl))

    t,pt = Get_derived_posterior(np.array(lm), dres)
    np.save(pos_path + '{0}_{1}_SFMfit_Plm'.format(field, galaxy),[t,pt])

#### gen line posteriors
if len(glob(pos_path + '{0}_{1}_SFMfit_Pline*'.format(field, galaxy))) == 0:
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0

    lines = []
    for i in range(len(MBS)):
        lines.append({})    
        for k in SF_temps:
            if k[0] == 'l':
                lines[i][k] = []

    for i in range(len(dres.samples)):
        m, a, m1, m2, m3, m4, m5, m6, d, z, s1, s2 = dres.samples[i]
        wave, flux = Gen_model(sp, [m, a, d], [m1, m2, m3, m4, m5, m6], agebins = 6, SF = True)

        Gchi2, fit = Fit_MB(MBS, [s1, s2], SF_temps, wave, flux, z)

        for ii in range(len(lines)):
            for k in lines[ii]:
                lines[ii][k].append(fit[ii]['cfit'][k][0])

    for i in range(len(lines)):
        for k in lines[i]:
            if sum(lines[i][k]) > 0:
                x,px = Get_derived_posterior(np.array(lines[i][k]), dres)
                np.save(pos_path + '{}_{}_SFMfit_P{}_{}'.format(field, galaxy, k, i),[x,px])

