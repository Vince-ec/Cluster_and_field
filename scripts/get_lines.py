#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import numpy as np
import fsps

if __name__ == '__main__':
    field = sys.argv[1] 
    galaxy = int(sys.argv[2])
    specz = float(sys.argv[3])


#full_db = pd.read_pickle(data_path + 'all_galaxies_1d.pkl')
full_db = pd.read_pickle('../dataframes/fitdb/all_galaxies_1d.pkl')

fit_db = full_db.query('field == "{}" and id == {}'.format(field,galaxy))

Gs = Gen_spec_2D(field, galaxy, specz)

Q_temps = Gen_temp_dict_addline(['Hd','Hg','Hb','Ha','OII','OIII','SII'])
lvals = [4102.89,4341.68,4862.68,6564.61,3727.092,5008.240,6718.29]

#############

def Q_spec_adjust(Gs):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)
    sp.params['dust2'] = fit_db['bfd'].values[0]
    sp.params['dust1'] = fit_db['bfd'].values[0]
    sp.params['logzsol'] = np.log10(fit_db['bfm'].values[0])
    
    time, sfr, tmax = convert_sfh(get_agebins(fit_db['bfa'].values[0]), 
        [fit_db['bfm1'].values[0], fit_db['bfm2'].values[0], fit_db['bfm3'].values[0], 
         fit_db['bfm4'].values[0], fit_db['bfm5'].values[0], 
         fit_db['bfm6'].values[0],fit_db['bfm7'].values[0], fit_db['bfm8'].values[0], 
         fit_db['bfm9'].values[0], fit_db['bfm10'].values[0]], maxage = fit_db['bfa'].values[0]*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = fit_db['bfa'].values[0], peraa = True)
    flam = F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**fit_db['bflm'].values[0]
    
    return wave, flam

def SF_spec_adjust(Gs):
    sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 2)
    sp.params['dust1'] = 0
    sp.params['dust2'] = fit_db['bfd'].values[0]
    sp.params['logzsol'] = np.log10(fit_db['bfm'].values[0])

    time, sfr, tmax = convert_sfh(get_agebins(fit_db['bfa'].values[0], binnum = 6), 
        [fit_db['bfm1'].values[0], fit_db['bfm2'].values[0], fit_db['bfm3'].values[0], fit_db['bfm4'].values[0], fit_db['bfm5'].values[0], 
         fit_db['bfm6'].values[0]], maxage = fit_db['bfa'].values[0]*1E9)

    sp.set_tabular_sfh(time,sfr) 

    wave, flux = sp.get_spectrum(tage = fit_db['bfa'].values[0], peraa = True)
    flam = F_lam_per_M(flux,wave*(1 + specz), specz, 0, sp.stellar_mass)*10**fit_db['bflm'].values[0]
    
    return wave, flam

######## Q-method
if fit_db.bfm10.values[0]**2 > 0:
    wave,flam = Q_spec_adjust(Gs)

######## SF-method
else:
    wave,flam = SF_spec_adjust(Gs)

Q_temps['fsps_model'] = SpectrumTemplate(wave=wave, flux=flam)


if Gs.g102:
    g102_fit = Gs.mb_g102.template_at_z(specz, templates = Q_temps, fitter='nnls')
if Gs.g141:
    g141_fit = Gs.mb_g141.template_at_z(specz, templates = Q_temps, fitter='nnls')

if Gs.g102:
    cfit_keys = list(g102_fit['cfit'].keys())
else:
    cfit_keys = list(g141_fit['cfit'].keys())

    
lines_db = {}

idx = 0
for i in range(len(cfit_keys)):
    lname = cfit_keys[i]
    if lname[0] == 'l':
        vals = []
        sigs = []
        
        if Gs.g102:
            val, sig = g102_fit['cfit'][lname] 
            if val > 0:
                vals.append(val)
                sigs.append(sig)
        if Gs.g141:
            val, sig = g141_fit['cfit'][lname]
            if val > 0:
                vals.append(val)
                sigs.append(sig)  
            
        if len(vals) > 1:
            if lvals[idx] * (Gs.specz + 1) < 11300:
                lines_db[lname.strip('line ')] = vals[0]
                lines_db[lname.strip('line ') + '_sig'] = sigs[0]
            else:
                lines_db[lname.strip('line ')] = vals[1]
                lines_db[lname.strip('line ') + '_sig'] = sigs[1]
                
        if len(vals) == 1:
            lines_db[lname.strip('line ')] = vals[0]
            lines_db[lname.strip('line ') + '_sig'] = sigs[0]
            
        if len(vals) == 0:
            lines_db[lname.strip('line ')] = -99
            lines_db[lname.strip('line ') + '_sig'] = -99
            
        idx += 1
print(lines_db)
#np.save(pos_path + '{}_{}_line_fits'.format(field, galaxy),lines_db, allow_pickle = True)