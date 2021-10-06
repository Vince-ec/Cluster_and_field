#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import numpy as np
import fsps
# import matplotlib.pyplot as plt
if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    SF = sys.argv[3]
    specz = float(sys.argv[4])

############
from spec_id import Calibrate_grism, Scale_model
def Best_fit_scale(wv, fl, er, mfl, p1):
    cal = Calibrate_grism([wv, fl, er], mfl, p1)
    scale = Scale_model(fl / cal, er/ cal, mfl)
    FL =  fl/ cal/ scale
    ER =  er/ cal/ scale
    return FL, ER

def Q_spec_adjust(Gs):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    sp.params['dust2'] = fit_db['bfd']
    sp.params['dust1'] = fit_db['bfd']
    sp.params['logzsol'] = np.log10(fit_db['bfm'])
    
    time, sfr, tmax = convert_sfh(get_agebins(fit_db['bfa']), 
        [fit_db['bfm1'], fit_db['bfm2'], fit_db['bfm3'], 
         fit_db['bfm4'], fit_db['bfm5'], 
         fit_db['bfm6'],fit_db['bfm7'], fit_db['bfm8'], 
         fit_db['bfm9'], fit_db['bfm10']], maxage = fit_db['bfa']*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = fit_db['bfa'], peraa = True)
    flam = F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**fit_db['bflm']
    
    return wave, flam, sp

def SF_spec_adjust(Gs):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    sp.params['dust2'] = fit_db['bfd']
    sp.params['logzsol'] = np.log10(fit_db['bfm'])
    
    time, sfr, tmax = convert_sfh(get_agebins(fit_db['bfa'], binnum = 6), 
        [fit_db['bfm1'], fit_db['bfm2'], fit_db['bfm3'], fit_db['bfm4'], fit_db['bfm5'], 
         fit_db['bfm6']], maxage = fit_db['bfa']*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = fit_db['bfa'], peraa = True)
    flam = F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**fit_db['bflm']
    
    return wave, flam, sp


#full_db = pd.read_pickle(data_path + 'all_galaxies_1d.pkl')
# full_db = pd.read_pickle(data_path + 'evolution_db_masslim.pkl')


# fields = ['GSD', 'GSD', 'GSD', 'GSD', 'GSD', 'GSD', 'GSD', 'GSD', 'GSD', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND', 'GND'] 
# galaxies = [26087, 29632, 29730, 27565, 28995, 30871, 36451, 37939, 46846, 9128, 11630, 13970, 14516, 16484, 18815, 18939, 19558, 20466, 20513, 20702, 21048, 23936, 27567, 27660, 32842, 33707, 35292, 37647] 


# for i in range(len(fields)):
#     field = fields[i]
#     galaxy = galaxies[i]

# print(full_db)
if SF == 'Q':
    fit_db = np.load(pos_path + '{}_{}_tabfit.npy'.format(field, galaxy), allow_pickle = True).item()
    specz = fit_db['bfz']
else:
    fit_db = np.load(pos_path + '{}_{}_SFfit_p1_fits.npy'.format(field, galaxy), allow_pickle = True).item()
    
# print(fit_db)

Gs = Gen_spec_2D(field, galaxy, specz)

######## Q-method
# if fit_db.bfm10.values[0]**2 > 0:
if SF == 'Q':
    wave, flam, sp = Q_spec_adjust(Gs)

######## SF-method
else:
    wave, flam, sp = SF_spec_adjust(Gs)
wvs, flxs, errs, beams, trans = Gather_grism_data_from_2d(Gs, sp)

Gmfl, Pmfl = Full_forward_model(Gs, wave,flam, specz, wvs, flxs, errs, beams, trans)

Bi = 'none'
Ri = 'none'
if Gs.g102:
    Bi = 0
else:
    Ri = 0

if Gs.g141 and Gs.g102:
    Ri = 1

if Gs.g102:
    BFL, BER = Best_fit_scale(wvs[Bi], flxs[Bi], errs[Bi], Gmfl[Bi],  fit_db['bfbp1'])

if Gs.g141:
    RFL, RER = Best_fit_scale(wvs[Ri], flxs[Ri], errs[Ri], Gmfl[Ri],  fit_db['bfrp1'])

spec_dict = {'Bwv':[],'Bfl':[], 'Ber':[], 'Bmfl':[], 
             'Rwv':[],'Rfl':[], 'Rer':[], 'Rmfl':[],       
             'Pwv':Gs.Pwv,'Pfl':Gs.Pflx, 'Per':Gs.Perr,
            'wave':wave, 'flam':flam}

if Gs.g102:
    spec_dict['Bwv'] = Gs.Bwv
    spec_dict['Bfl'] = BFL
    spec_dict['Ber'] = BER
    spec_dict['Bmfl'] =  Gmfl[Bi]

if Gs.g141:
    spec_dict['Rwv'] = Gs.Rwv
    spec_dict['Rfl'] = RFL
    spec_dict['Rer'] = RER
    spec_dict['Rmfl'] = Gmfl[Ri]

np.save(pos_path + '{}_{}_fullspec'.format(field, galaxy), spec_dict, allow_pickle = True)
# np.save('../full_specs/{}_{}_fullspec'.format(field, galaxy), spec_dict, allow_pickle = True)
