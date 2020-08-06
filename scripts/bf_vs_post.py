import fsps
import numpy as np
from spec_id import *
import matplotlib.pyplot as plt
import pandas as pd
from spec_tools import Posterior_spec

alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
morph_db = alldb.query('W_UVJ == "Q" and AGN != "AGN" and lmass >= 10.5 and n_f < 3 and Re < 20 ')

bspec = [27458,294464,36348,48631,19290,32566,32691,33093,26272,35640,45333, 30144, 21683]
# nog141 = [27915,37955,17746,17735]
nog102 = [27714,37189,26139,32799,47223,22774,28890,23073,31452,24033]
# nog102 = []

inout = []
for i in morph_db.index:     
    if morph_db.id[i] not in bspec and morph_db.id[i] not in nog102: 
        inout.append('i')
    else:
        inout.append('o')
        
morph_db['inout'] = inout
mdb = morph_db.query('inout == "i" and 0.7 < zgrism < 2.5 and Sigma1 > 10**9.6')
#############################################
def Full_calibrate(mfl, p1, wv, fl, er):
    cal = Calibrate_grism([wv, fl, er], mfl, p1)
    scale = Scale_model( fl / cal, er/ cal,  mfl)
    nfl = fl/ cal/ scale
    ner = er/ cal/ scale
    return nfl, ner     
sp = fsps.StellarPopulation(zcontinuous = 1, logzsol = 0, sfh = 3, dust_type = 1)

for idx in mdb.index:
    Gs = Gen_spec(mdb.field[idx], mdb.id[idx], mdb.zgrism[idx],
                  g102_lims=[8300, 11288], g141_lims=[11288, 16500],phot_errterm = 0.04, irac_err = 0.08,) 

    bfm, bfa, bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10, bflm, bfz, bfd,\
    bfbp1, bfrp1, bfba, bfbb, bfbl, bfra, bfrb, bfrl, bflwa, logl =np.load(
        '../data/bestfits/{}_{}_tabfit_bfit.npy'.format(mdb.field[idx], mdb.id[idx]))

    #sp.params['dust2'] = bfd
    #sp.params['dust1'] = bfd
    #sp.params['logzsol'] = np.log10(bfm)
    
    #time, sfr, tmax = convert_sfh(get_agebins(bfa), [bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, bfm9, bfm10], maxage = bfa*1E9)

    #sp.set_tabular_sfh(time,sfr) 
    
    #wave, flux = sp.get_spectrum(tage = bfa, peraa = True)
    
    #flam = F_lam_per_M(flux,wave*(1+bfz), bfz, 0, sp.stellar_mass)*10**bflm
    
    #np.save('../data/allsed/phot/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]), [wave,flam])
    wave,flam = np.load('../data/allsed/phot/{}-{}_mod.npy'.format(mdb.field[idx], mdb.id[idx]))

    Gs.Sim_all_premade(wave*(1+mdb.zgrism[idx]), flam)

    if Gs.g102:
        nflx, nerr = Full_calibrate(Gs.Bmfl, bfbp1,Gs.Bwv, Gs.Bfl, Gs.Ber)
        np.save('../data/allsed/g102/{}-{}'.format(mdb.field[idx], mdb.id[idx]),[Gs.Bwv, nflx, nerr])
        np.save('../data/allsed/g102/{}-{}_mod'.format(mdb.field[idx], mdb.id[idx]),[Gs.Bwv, Gs.Bmfl])

    if Gs.g141:
        nflx, nerr = Full_calibrate(Gs.Rmfl, bfrp1,Gs.Rwv, Gs.Rfl, Gs.Rer)
        np.save('../data/allsed/g141/{}-{}'.format(mdb.field[idx], mdb.id[idx]),[Gs.Rwv, nflx, nerr])
        np.save('../data/allsed/g141/{}-{}_mod'.format(mdb.field[idx], mdb.id[idx]),[Gs.Rwv,Gs.Rmfl])

"""
for idx in mdb.index:

    plt.figure(figsize=[15,6])

    try:
        bwv, bfl, ber = np.load('../data/allsed/g102/{}-{}_O.npy'.format(mdb.field[idx], mdb.id[idx]))
        plt.errorbar(bwv, bfl, ber, color = 'k')
    except:
        pass

    try:
        rwv, rfl, rer = np.load('../data/allsed/g141/{}-{}_O.npy'.format(mdb.field[idx], mdb.id[idx]))
        plt.errorbar(rwv, rfl, rer, color = 'k')
    except:
        pass

    try:
        lims = [8300, 11288]
        W, F, E, FLT, L, C = np.load(spec_path + '{}_{}_g102.npy'.format(mdb.field[idx], mdb.id[idx]),allow_pickle=True)
        IDX = [U for U in range(len(W)) if lims[0] <= W[U] <= lims[-1] and F[U]**2 > 0]
        W = np.array(W[IDX])
        F = np.array(F[IDX])
        FLT = np.array(FLT[IDX]) 
        E = np.array(E[IDX])  
        plt.errorbar(W,F/FLT,E/FLT, color = 'r')
    except:
        pass

    try:
        lims=[11288, 16500]
        W, F, E, FLT, L, C = np.load(spec_path + '{}_{}_g141.npy'.format(mdb.field[idx], mdb.id[idx]),allow_pickle=True)
        IDX = [U for U in range(len(W)) if lims[0] <= W[U] <= lims[-1] and F[U]**2 > 0]
        W = np.array(W[IDX])
        F = np.array(F[IDX]) 
        FLT = np.array(FLT[IDX]) 
        E = np.array(E[IDX])  
        plt.errorbar(W,F/FLT,E/FLT, color = 'r')
    except:
        pass

    plt.title('{}-{}'.format(mdb.field[idx], mdb.id[idx]))
    plt.savefig('../plots/bf_v_pos/{}-{}.png'.format(mdb.field[idx], mdb.id[idx]), bbox_inches = 'tight') """   