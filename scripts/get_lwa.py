#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import numpy as np
import fsps

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])

dres = np.load(out_path + '{0}_{1}_SFfit_p1.npy'.format(field, galaxy),allow_pickle=True).item()

sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
sp.params['dust1'] = 0
  
    
def get_lwa_SF_u(params, agebins,sp):
    m, a, m1, m2, m3, m4, m5, m6 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_u'])
    sp.params['compute_light_ages'] = False
    
    return lwa

def get_lwa_SF_r(params, agebins,sp):
    m, a, m1, m2, m3, m4, m5, m6 = params

    sp.params['logzsol'] = np.log10(m)

    time, sfr, tmax = convert_sfh(agebins, [m1, m2, m3, m4, m5, m6])

    sp.set_tabular_sfh(time,sfr)    
    
    sp.params['compute_light_ages'] = True
    lwa = sp.get_mags(tage = a, bands=['sdss_r'])
    sp.params['compute_light_ages'] = False
    
    return lwa
    
    
lwa = []
lwa_u = []
lwa_r = []

for i in range(len(dres.samples)):
    m, a, m1, m2, m3, m4, m5, m6, lm, d, bp1, rp1, ba, bb, bl, ra, rb, rl = dres.samples[i]
    lwa.append(get_lwa_SF([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum=6),sp)[0])
    lwa_u.append(get_lwa_SF_u([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum=6),sp)[0])
    lwa_r.append(get_lwa_SF_r([m, a, m1, m2, m3, m4, m5, m6], get_agebins(a, binnum=6),sp)[0])

g,pg = Get_derived_posterior(np.array(lwa), dres)
u,pu = Get_derived_posterior(np.array(lwa_u), dres)
r,pr = Get_derived_posterior(np.array(lwa_r), dres)

fit_db = np.load(pos_path + '{}_{}_SFfit_p1_fits.npy'.format(field, galaxy), allow_pickle=True).item()

fit_db['Plwa'] = pg
fit_db['lwa'] = g

fit_db['Plwa_u'] = pu
fit_db['lwa_u'] = u

fit_db['Plwa_r'] = pr
fit_db['lwa_r'] = r

np.save(pos_path + '{}_{}_SFfit_p1_fits'.format(field, galaxy),fit_db)