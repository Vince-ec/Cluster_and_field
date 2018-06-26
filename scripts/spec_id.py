__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
import sympy as sp
import grizli
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import os
from glob import glob

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()

def Oldest_galaxy(z):
    return cosmo.age(z).value


def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return G / np.trapz(G, x)

def Median_w_Error(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_95(Pofx, x):
    iP = interp1d(x, Pofx)
    ix = np.linspace(x[0], x[-1], 500)

    lerr = 0
    herr = 0

    for i in range(len(ix)):
        e = np.trapz(iP(ix[0:i + 1]), ix[0:i + 1])
        if lerr == 0:
            if e >= .025:
                lerr = ix[i]
        if herr == 0:
            if e >= .975:
                herr = ix[i]
                break

    med = 0

    for i in range(len(x)):
        e = np.trapz(Pofx[0:i + 1], x[0:i + 1])
        if med == 0:
            if e >= .5:
                med = x[i]
                break

    return np.round(med,3), np.round(med - lerr,3), np.round(herr - med,3)

def Median_w_Error_cont(Pofx, x):
    ix = np.linspace(x[0], x[-1], 500)
    iP = interp1d(x, Pofx)(ix)

    C = np.trapz(iP,ix)

    iP/=C


    lerr = 0
    herr = 0
    med = 0

    for i in range(len(ix)):
        e = np.trapz(iP[0:i + 1], ix[0:i + 1])
        if lerr == 0:
            if e >= .16:
                lerr = ix[i]
        if med == 0:
            if e >= .50:
                med = ix[i]
        if herr == 0:
            if e >= .84:
                herr = ix[i]
                break

    return med, med - lerr, herr - np.abs(med)

def Scale_model(D, sig, M):
    return np.sum(((D * M) / sig ** 2)) / np.sum((M ** 2 / sig ** 2))

def Likelihood_contours(age, metallicty, prob):
    ####### Create fine resolution ages and metallicities
    ####### to integrate over
    m2 = np.linspace(min(metallicty), max(metallicty), 50)

    ####### Interpolate prob
    P2 = interp2d(metallicty, age, prob)(m2, age)

    ####### Create array from highest value of P2 to 0
    pbin = np.linspace(0, np.max(P2), 1000)
    pbin = pbin[::-1]

    ####### 2d integrate to find the 1 and 2 sigma values
    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(P2)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(np.trapz(p, m2, axis=1), age)

    ######## Identify 1 and 2 sigma values
    onesig = np.abs(np.array(prob_int) - 0.68)
    twosig = np.abs(np.array(prob_int) - 0.95)

    return pbin[np.argmin(onesig)], pbin[np.argmin(twosig)]


"""Single Galaxy"""
class Gen_spec(object):
    def __init__(self, galaxy_id, redshift, pad=100, delayed = True,minwv = 7900, maxwv = 11300):
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.pad = pad
        self.delayed = delayed

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        if self.galaxy_id == 's35774':
            maxwv = 11100

        gal_wv, gal_fl, gal_er = np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S
        model = '../../../fsps_models_for_fit/fsps_spec/m{0}_a{1}_dt{2}_spec.npy'.format(metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

    def Fit_lwa(self, fit_Z, fit_t, metal_array, age_array, tau_array):
        
        lwa_grid = np.load('../data/light_weight_scaling_3.npy')
        chi = []
        good_age =[]
        good_tau =[]
        for i in range(len(tau_array)):
            for ii in range(age_array.size):
                
                lwa = lwa_grid[np.argwhere(np.round(metal_array,3) == np.round(fit_Z,3))[0][0]][ii][i]
                
                if (fit_t - 0.1) < lwa < (fit_t + 0.1):
                    self.Sim_spec(fit_Z,age_array[ii],tau_array[i])
                    chi.append(sum(((self.gal_fl - self.fl) / self.gal_er)**2))
                    good_age.append(age_array[ii])
                    good_tau.append(tau_array[i])

        self.bfage = np.array(good_age)[chi == min(chi)][0]
        self.bftau = np.array(good_tau)[chi == min(chi)][0]
        if self.bftau == 0.0:
            self.bftau = int(0)
        self.Sim_spec(fit_Z, self.bfage, self.bftau)    


def Single_gal_fit_full(metal, age, tau, specz, galaxy, name, minwv = 7900, maxwv = 11300):
    #############Read in spectra#################
    spec = Gen_spec(galaxy, specz, minwv = minwv, maxwv = maxwv)

    if galaxy == 'n21156' or galaxy == 'n38126':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4855 <= spec.gal_wv_rf[ii] <= 4880:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's47677' or galaxy == 'n14713':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4845 <= spec.gal_wv_rf[ii] <= 4863:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    if galaxy == 's39170':
        IDer = []
        for ii in range(len(spec.gal_wv_rf)):
            if 4865 <= spec.gal_wv_rf[ii] <= 4885:
                IDer.append(ii)
        spec.gal_er[IDer] = 1E8
        spec.gal_fl[IDer] = 0

    IDF = []
    for i in range(len(spec.gal_wv_rf)):
        if 3800 <= spec.gal_wv_rf[i] <= 3850 or 3910 <= spec.gal_wv_rf[i] <= 4030 or 4080 <= spec.gal_wv_rf[i] <= 4125 \
                or 4250 <= spec.gal_wv_rf[i] <= 4385 or 4515 <= spec.gal_wv_rf[i] <= 4570 or 4810 <= spec.gal_wv_rf[i]\
                <= 4910 or 4975 <= spec.gal_wv_rf[i] <= 5055 or 5110 <= spec.gal_wv_rf[i] <= 5285:
            IDF.append(i)

    IDC = []
    for i in range(len(spec.gal_wv_rf)):
        if spec.gal_wv_rf[0] <= spec.gal_wv_rf[i] <= 3800 or 3850 <= spec.gal_wv_rf[i] <= 3910 or 4030 <= \
                spec.gal_wv_rf[i] <= 4080 or 4125 <= spec.gal_wv_rf[i] <= 4250 or 4385 <= spec.gal_wv_rf[i] <= 4515 or \
                4570 <= spec.gal_wv_rf[i] <= 4810 or 4910 <= spec.gal_wv_rf[i] <= 4975 or 5055 <= spec.gal_wv_rf[i] <= \
                5110 or 5285 <= spec.gal_wv_rf[i] <= spec.gal_wv_rf[-1]:
            IDC.append(i)

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1 = '../chidat/%s_chidata' % name
    chifile2 = '../chidat/%s_cont_chidata' % name
    chifile3 = '../chidat/%s_feat_chidata' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(tau),len(spec.gal_wv_rf)])
    mfl_f = np.zeros([len(metal)*len(age)*len(tau),len(IDF)])
    mfl_c = np.zeros([len(metal)*len(age)*len(tau),len(IDC)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                mfl[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl
                mfl_f[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl[IDF]
                mfl_c[i*len(age)*len(tau)+ii*len(tau)+iii]=spec.fl[IDC]
    chigrid1 = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)
    chigrid2 = np.sum(((spec.gal_fl[IDF] - mfl_f) / spec.gal_er[IDF]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)
    chigrid3 = np.sum(((spec.gal_fl[IDC] - mfl_c) / spec.gal_er[IDC]) ** 2, axis=1).reshape([len(metal), len(age), len(tau)]).\
        astype(np.float128)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)
    np.save(chifile2,chigrid2)
    np.save(chifile3,chigrid3)

    P, PZ, Pt = Analyze_LH_cont_feat(chifile2 + '.npy', chifile3 + '.npy', specz, metal, age, tau)

    np.save('../chidat/%s_tZ_pos' % name,P)
    np.save('../chidat/%s_Z_pos' % name,[metal,PZ])
    np.save('../chidat/%s_t_pos' % name,[age,Pt])

    print 'Done!'
    return


def Specz_fit(galaxy, metal, age, rshift, name):
    #############initialize spectra#################
    spec = RT_spec(galaxy)

    #############Prep output file###############
    chifile = '../rshift_dat/%s_z_fit' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(rshift),len(spec.gal_wv)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(rshift)):
                spec.Sim_spec(metal[i], age[ii], 0, rshift[iii])
                mfl[i*len(age)*len(rshift)+ii*len(rshift)+iii]=spec.fl
    chigrid = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(rshift)]).\
        astype(np.float128)

    np.save(chifile,chigrid)
    ###############Write chigrid file###############
    Analyze_specz(chifile + '.npy', rshift, metal, age, name)

    print 'Done!'

    return


def Norm_P_specz(rshift, metal, age, chi):
    ####### Heirarchy is rshift_-> age -> metal
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    prob = []
    for i in range(len(rshift)):
        preprob1 = []
        for ii in range(len(age)):
            preprob2 = []
            for iii in range(len(metal)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii] / 2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all metal
    ######## End up with age vs rshift matricies
    R = []
    for i in range(len(rshift)):
        A = []
        for ii in range(len(age)):
            M = []
            for iii in range(len(metal) - 1):
                M.append(sp.N((metal[iii + 1] - metal[iii]) * (prob[i][ii][iii] + prob[i][ii][iii + 1]) / 2))
            A.append(sp.mpmath.fsum(M))
        R.append(A)

    ######## Integrate over age to get rshift prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(rshift)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (R[i][ii] + R[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(rshift) - 1):
        preC3.append(sp.N((rshift[i + 1] - rshift[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    ######## Create normal prob grid
    P = []
    for i in range(len(rshift)):
        P.append(preC1[i] / C)

    return np.array(P).astype(np.float128)


def Analyze_specz(chifits, rshift, metal, age, name):
    ####### Read in file
    dat = np.load(chifits)

    ###### Create normalize probablity marginalized over tau
    prob = np.array(Norm_P_specz(rshift, metal, age, dat.T)).astype(np.float128)

    ###### get best fit values
    print 'Best fit specz is %s' % rshift[np.argmax(prob)]

    np.save('../rshift_dat/%s_Pofz' % name,[rshift, prob])
    return


def Analyze_LH_lwa(chifits, specz, metal, age, tau, age_conv='../data/light_weight_scaling_3.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E5

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        # if i == 0:
        #     newchi[i] = chi[i]
        # else:
        frame = np.zeros([metal.size,age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii],chi[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist,np.repeat(1E5, overhead[i][ii]))
        newchi[i] = frame.T

    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt


class Galaxy_set(object):
    def __init__(self, galaxy_id):
        self.galaxy_id = galaxy_id
        if os.path.isdir('../../../../vestrada'):
            gal_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s/' % self.galaxy_id
        else:
            gal_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s/' % self.galaxy_id

        # test
        # gal_dir = '/Users/Vince.ec/Clear_data/test_data/%s/' % self.galaxy_id
        one_d = glob(gal_dir + '*1D.fits')
        self.two_d = glob(gal_dir + '*png')
        one_d_l = [len(U) for U in one_d]
        self.one_d_stack = one_d[np.argmin(one_d_l)]
        self.one_d_list = np.delete(one_d, [np.argmin(one_d_l)])

    def Get_flux(self, FILE):
        observ = fits.open(FILE)
        w = np.array(observ[1].data.field('wave'))
        f = np.array(observ[1].data.field('flux')) * 1E-17
        sens = np.array(observ[1].data.field('sensitivity'))
        contam = np.array(observ[1].data.field('contam')) * 1E-17
        e = np.array(observ[1].data.field('error')) * 1E-17
        f -= contam
        f /= sens
        e /= sens

        INDEX = []
        for i in range(len(w)):
            if w[i] < 11900:
                INDEX.append(i)

        w = w[INDEX]
        f = f[INDEX]
        e = e[INDEX]

        # for i in range(len(f)):
        #     if f[i] < 0:
        #         f[i] = 0

        return w, f, e

    def Display_spec(self, override_quality=False):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            if override_quality == True:

                if len(self.two_d) > 0:
                    for i in range(len(self.two_d)):
                        os.system("open " + self.two_d[i])

                if len(self.one_d_list) > 0:
                    if len(self.one_d_list) < 10:
                        plt.figure(figsize=[15, 10])
                        for i in range(len(self.one_d_list)):
                            wv, fl, er = Get_flux(self.one_d_list[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(self.one_d_list) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                    if len(self.one_d_list) > 10:

                        smlist1 = self.one_d_list[:9]
                        smlist2 = self.one_d_list[9:]

                        plt.figure(figsize=[15, 10])
                        for i in range(len(smlist1)):
                            wv, fl, er = Get_flux(smlist1[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(smlist1) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                        plt.figure(figsize=[15, 10])
                        for i in range(len(smlist2)):
                            wv, fl, er = Get_flux(smlist2[i])
                            IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                            plt.subplot(11 + i + len(smlist2) * 100)
                            plt.plot(wv[IDX], fl[IDX])
                            plt.plot(wv[IDX], er[IDX])
                            plt.ylim(min(fl[IDX]), max(fl[IDX]))
                            plt.xlim(7800, 11500)
                        plt.show()

                self.quality = np.repeat(1, len(self.one_d_list)).astype(int)
                self.Mask = np.zeros([len(self.one_d_list), 2])
                self.pa_names = []

                for i in range(len(self.one_d_list)):
                    self.pa_names.append(self.one_d_list[i].replace(n_dir, ''))
                    wv, fl, er = Get_flux(self.one_d_list[i])
                    IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                    plt.figure(figsize=[15, 5])
                    plt.plot(wv[IDX], fl[IDX])
                    plt.plot(wv[IDX], er[IDX])
                    plt.ylim(min(fl[IDX]), max(fl[IDX]))
                    plt.xlim(7800, 11500)
                    plt.title(self.pa_names[i])
                    plt.show()
                    self.quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
                    if self.quality[i] == 1:
                        minput = int(input('Mask region: (0 if no mask needed)'))
                        if minput != 0:
                            rinput = int(input('Lower bounds'))
                            linput = int(input('Upper bounds'))
                            self.Mask[i] = [rinput, linput]
                ### save quality file
                l_mask = self.Mask.T[0]
                h_mask = self.Mask.T[1]

                qual_dat = Table([self.pa_names, self.quality, l_mask, h_mask],
                                 names=['id', 'good_spec', 'mask_low', 'mask_high'])
                fn = n_dir + '/%s_quality.txt' % self.galaxy_id
                ascii.write(qual_dat, fn, overwrite=True)

        else:

            if len(self.two_d) > 0:
                for i in range(len(self.two_d)):
                    os.system("open " + self.two_d[i])

            if len(self.one_d_list) > 0:
                if len(self.one_d_list) < 10:
                    plt.figure(figsize=[15, 10])
                    for i in range(len(self.one_d_list)):
                        wv, fl, er = Get_flux(self.one_d_list[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11750]
                        plt.subplot(11 + i + len(self.one_d_list) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11750)
                    plt.show()

                if len(self.one_d_list) > 10:

                    smlist1 = self.one_d_list[:9]
                    smlist2 = self.one_d_list[9:]

                    plt.figure(figsize=[15, 10])
                    for i in range(len(smlist1)):
                        wv, fl, er = Get_flux(smlist1[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                        plt.subplot(11 + i + len(smlist1) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11500)
                    plt.show()

                    plt.figure(figsize=[15, 10])
                    for i in range(len(smlist2)):
                        wv, fl, er = Get_flux(smlist2[i])
                        IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                        plt.subplot(11 + i + len(smlist2) * 100)
                        plt.plot(wv[IDX], fl[IDX])
                        plt.plot(wv[IDX], er[IDX])
                        plt.ylim(min(fl[IDX]), max(fl[IDX]))
                        plt.xlim(7800, 11500)
                    plt.show()

            self.quality = np.repeat(1, len(self.one_d_list)).astype(int)
            self.Mask = np.zeros([len(self.one_d_list), 2])
            self.pa_names = []

            for i in range(len(self.one_d_list)):
                self.pa_names.append(self.one_d_list[i].replace(n_dir, ''))
                wv, fl, er = Get_flux(self.one_d_list[i])
                IDX = [U for U in range(len(wv)) if 7700 <= wv[U] <= 11500]
                plt.figure(figsize=[15, 5])
                plt.plot(wv[IDX], fl[IDX])
                plt.plot(wv[IDX], er[IDX])
                plt.ylim(min(fl[IDX]), max(fl[IDX]))
                plt.xlim(7800, 11500)
                plt.title(self.pa_names[i])
                plt.show()
                self.quality[i] = int(input('Is this spectra good: (1 yes) (0 no)'))
                if self.quality[i] == 1:
                    minput = int(input('Mask region: (0 if no mask needed)'))
                    if minput != 0:
                        rinput = int(input('Lower bounds'))
                        linput = int(input('Upper bounds'))
                        self.Mask[i] = [rinput, linput]
            ### save quality file
            l_mask = self.Mask.T[0]
            h_mask = self.Mask.T[1]

            qual_dat = Table([self.pa_names, self.quality, l_mask, h_mask],
                             names=['id', 'good_spec', 'mask_low', 'mask_high'])
            fn = n_dir + '/%s_quality.txt' % self.galaxy_id
            ascii.write(qual_dat, fn, overwrite=True)

    def Get_wv_list(self):
        W = []
        lW = []

        for i in range(len(self.one_d_list)):
            wv, fl, er = self.Get_flux(self.one_d_list[i])
            W.append(wv)
            lW.append(len(wv))

        W = np.array(W)
        self.wv = W[np.argmax(lW)]

    def Mean_stack_galaxy(self):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        ### select good galaxies
        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            self.pa_names, self.quality, l_mask, h_mask = Readfile(n_dir + '/%s_quality.txt' % self.galaxy_id,
                                                                   is_float=False)
            self.quality = self.quality.astype(float)
            l_mask = l_mask.astype(float)
            h_mask = h_mask.astype(float)
            self.Mask = np.vstack([l_mask, h_mask]).T

        new_speclist = []
        new_mask = []
        for i in range(len(self.quality)):
            if self.quality[i] == 1:
                new_speclist.append(self.one_d_list[i])
                new_mask.append(self.Mask[i])

        self.Get_wv_list()
        self.good_specs = new_speclist
        self.good_Mask = new_mask

        # Define grids used for stacking
        flgrid = np.zeros([len(self.good_specs), len(self.wv)])
        errgrid = np.zeros([len(self.good_specs), len(self.wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(self.good_specs)):
            wave, flux, error = self.Get_flux(self.good_specs[i])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv])
            ifl = interp1d(wave, flux)(self.wv[mask])
            ier = interp1d(wave, error)(self.wv[mask])

            if sum(self.good_Mask[i]) > 0:
                for ii in range(len(self.wv[mask])):
                    if self.good_Mask[i][0] < self.wv[mask][ii] < self.good_Mask[i][1]:
                        ifl[ii] = 0
                        ier[ii] = 0

            flgrid[i][mask] = ifl
            errgrid[i][mask] = ier

        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv)])
        for i in range(len(self.wv)):
            # fl_filter = np.ones(len(flgrid[i]))
            # for ii in range(len(flgrid[i])):
            #     if flgrid[i][ii] == 0:
            #         fl_filter[ii] = 0
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.fl = np.array(stack)
        self.er = np.array(err)

    def Bootstrap(self, galaxy_list, repeats=1000):
        gal_index = np.arange(len(galaxy_list))

    def Median_w_bootstrap_stack_galaxy(self):
        if os.path.isdir('../../../../vestrada'):
            n_dir = '../../../../../Volumes/Vince_research/Extractions/Quiescent_galaxies/%s' % self.galaxy_id
        else:
            n_dir = '../../../../../Volumes/Vince_homedrive/Extractions/Quiescent_galaxies/%s' % self.galaxy_id

        ### select good galaxies
        if os.path.isfile(n_dir + '/%s_quality.txt' % self.galaxy_id):
            self.pa_names, self.quality, l_mask, h_mask = Readfile(n_dir + '/%s_quality.txt' % self.galaxy_id,
                                                                   is_float=False)
            self.quality = self.quality.astype(float)
            l_mask = l_mask.astype(float)
            h_mask = h_mask.astype(float)
            self.Mask = np.vstack([l_mask, h_mask]).T

        new_speclist = []
        new_mask = []
        for i in range(len(self.quality)):
            if self.quality[i] == 1:
                new_speclist.append(self.one_d_list[i])
                new_mask.append(self.Mask[i])

        self.Get_wv_list()
        self.good_specs = new_speclist
        self.good_Mask = new_mask

        # Define grids used for stacking
        flgrid = np.zeros([len(self.good_specs), len(self.wv)])
        errgrid = np.zeros([len(self.good_specs), len(self.wv)])

        # Get wv,fl,er for each spectra
        for i in range(len(self.good_specs)):
            wave, flux, error = self.Get_flux(self.good_specs[i])
            mask = np.array([wave[0] < U < wave[-1] for U in self.wv])
            ifl = interp1d(wave, flux)(self.wv[mask])
            ier = interp1d(wave, error)(self.wv[mask])

            if sum(self.good_Mask[i]) > 0:
                for ii in range(len(self.wv[mask])):
                    if self.good_Mask[i][0] < self.wv[mask][ii] < self.good_Mask[i][1]:
                        ifl[ii] = 0
                        ier[ii] = 0

            flgrid[i][mask] = ifl
            errgrid[i][mask] = ier

        ################

        flgrid = np.transpose(flgrid)
        errgrid = np.transpose(errgrid)
        weigrid = errgrid ** (-2)
        infmask = np.isinf(weigrid)
        weigrid[infmask] = 0
        ################

        stack, err = np.zeros([2, len(self.wv)])
        for i in range(len(self.wv)):
            # fl_filter = np.ones(len(flgrid[i]))
            # for ii in range(len(flgrid[i])):
            #     if flgrid[i][ii] == 0:
            #         fl_filter[ii] = 0
            stack[i] = np.sum(flgrid[i] * weigrid[[i]]) / (np.sum(weigrid[i]))
            err[i] = 1 / np.sqrt(np.sum(weigrid[i]))
        ################

        self.fl = np.array(stack)
        self.er = np.array(err)

    def Get_stack_info(self):
        wv, fl, er = Get_flux(self.one_d_stack)
        self.s_wv = wv
        self.s_fl = fl
        self.s_er = er


"""MC fits"""


class Gen_sim(object):
    def __init__(self, galaxy_id, redshift, metal, age, tau, minwv=7900, maxwv=11400, pad=100):
        import pysynphot as S
        self.galaxy_id = galaxy_id
        self.redshift = redshift
        self.metal = metal
        self.age = age
        self.tau = tau
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.gal_wv - output wavelength array of galaxy
        **
        self.gal_wv_rf - output wavelength array in restframe
        **
        self.gal_fl - output flux array of galaxy
        **
        self.gal_er - output error array of galaxy
        **
        self.fl - output flux array of model used for simulation
        **
        self.flx_err - output flux array of model perturb by the galaxy's 1 sigma errors
        **
        self.mfl - output flux array of model generated to fit against 
        """

        gal_wv, gal_fl, gal_er = np.load('../spec_stacks_june14/%s_stack.npy' % self.galaxy_id)
        self.flt_input = '../data/galaxy_flts/%s_flt.fits' % self.galaxy_id

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + self.redshift)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0]
        self.gal_wv = self.gal_wv[self.gal_fl > 0]
        self.gal_er = self.gal_er[self.gal_fl > 0]
        self.gal_fl = self.gal_fl[self.gal_fl > 0]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

        ## create basis model for sim

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (self.metal, self.age, self.tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl

    def Perturb_flux(self):
        self.flx_err = np.abs(self.fl + np.random.normal(0, self.gal_er))

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S

        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl / filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.mfl = C * adj_ifl


"""Test Functions"""


def Best_fit_model(input_file, metal, age, tau):
    dat = fits.open(input_file)

    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    x = np.argwhere(chi == np.min(chi))
    print metal[x[0][0]], age[x[0][1]], tau[x[0][2]]
    return metal[x[0][0]], age[x[0][1]], tau[x[0][2]]

def B_factor(input_chi_file, tau, metal, age):
    ####### Heirarchy is metallicity_-> age -> tau
    ####### Change chi to probabilites using sympy
    ####### for its arbitrary precission, must be done in loop
    dat = fits.open(input_chi_file)
    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    prob = []
    for i in range(len(metal)):
        preprob1 = []
        for ii in range(len(age)):
            preprob2 = []
            for iii in range(len(tau)):
                preprob2.append(sp.N(sp.exp(-chi[i][ii][iii] / 2)))
            preprob1.append(preprob2)
        prob.append(preprob1)

    ######## Marginalize over all tau
    ######## End up with age vs metallicity matricies
    ######## use unlogged tau
    ultau = np.append(0, np.power(10, tau[1:] - 9))
    M = []
    for i in range(len(metal)):
        A = []
        for ii in range(len(age)):
            T = []
            for iii in range(len(tau) - 1):
                T.append(sp.N((ultau[iii + 1] - ultau[iii]) * (prob[i][ii][iii] + prob[i][ii][iii + 1]) / 2))
            A.append(sp.mpmath.fsum(T))
        M.append(A)

    ######## Integrate over metallicity to get age prob
    ######## Then again over age to find normalizing coefficient
    preC1 = []
    for i in range(len(metal)):
        preC2 = []
        for ii in range(len(age) - 1):
            preC2.append(sp.N((age[ii + 1] - age[ii]) * (M[i][ii] + M[i][ii + 1]) / 2))
        preC1.append(sp.mpmath.fsum(preC2))

    preC3 = []
    for i in range(len(metal) - 1):
        preC3.append(sp.N((metal[i + 1] - metal[i]) * (preC1[i] + preC1[i + 1]) / 2))

    C = sp.mpmath.fsum(preC3)

    return C



#####STATS

def Leave_one_out(dist, x):
    Y = np.zeros(x.size)
    for i in range(len(dist)):
        Y += dist[i]
    Y /= np.trapz(Y, x)

    w = np.arange(.01, 2.01, .01)
    weights = np.zeros(len(dist))
    for i in range(len(dist)):
        Ybar = np.zeros(x.size)
        for ii in range(len(dist)):
            if i != ii:
                Ybar += dist[ii]
        Ybar /= np.trapz(Ybar, x)
        weights[i] = np.sum((Ybar - Y) ** 2) ** -1
    return weights

def Stack_posteriors(P_grid, x):
    P_grid = np.array(P_grid)
    W = Leave_one_out(P_grid,x)
    top = np.zeros(P_grid.shape)
    for i in range(W.size):
        top[i] = W[i] * P_grid[i]
    P =sum(top)/sum(W)
    return P / np.trapz(P,x)

def Iterative_stacking(grid_o,x_o,iterations = 20,resampling = 250):
    ksmooth = importr('KernSmooth')
    del_x = x_o[1] - x_o[0]

    ### resample
    x = np.linspace(x_o[0],x_o[-1],resampling)
    grid = np.zeros([len(grid_o),x.size])    
    for i in range(len(grid_o)):
        grid[i] = interp1d(x_o,grid_o[i])(x)
   
    ### select bandwidth
    H = ksmooth.dpik(x)
    ### stack posteriors w/ weights
    stkpos = Stack_posteriors(grid,x)
    ### initialize prior as flat
    Fx = np.ones(stkpos.size)
    
    for i in range(iterations):
        fnew = Fx * stkpos / np.trapz(Fx * stkpos,x)
        fx = ksmooth.locpoly(x,fnew,bandwidth = H)
        X = np.array(fx[0])
        iFX = np.array(fx[1])
        Fx = interp1d(X,iFX)(x)

    Fx[Fx<0]=0
    Fx = Fx/np.trapz(Fx,x)
    return Fx,x

def Linear_fit(x,Y,sig,new_x,return_cov = False):
    A=np.array([np.ones(len(x)),x]).T
    C =np.diag(sig**2)
    iC=inv(C)
    b,m = np.dot(inv(np.dot(np.dot(A.T,iC),A)),np.dot(np.dot(A.T,iC),Y))
    cov = inv(np.dot(np.dot(A.T,iC),A))
    var_b = cov[0][0]
    var_m = cov[1][1]
    sig_mb = cov[0][1]
    sig_y = np.sqrt(var_b + new_x**2*var_m + 2*new_x*sig_mb)
    if return_cov == True:
        return m*new_x+b , sig_y, cov
    else:
        return m*new_x+b , sig_y

def Gen_grid(DB,param):
    grid=[]
    for i in DB.index:
        x,Px = np.load('../chidat/%s_dtau_%s_pos_lwa_3.npy' % (DB['gids'][i],param))
        grid.append(Px)
    return np.array(grid)

"""Proposal fit 2D"""

class Gen_spec_2d(object):
    def __init__(self, stack_2d, stack_2d_error, grism_flt, direct_flt, redshift):
        self.stack_2d = stack_2d
        self.stack_2d_error = stack_2d_error
        self.grism = grism_flt
        self.direct = direct_flt
        self.redshift = redshift

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """

        self.gal = np.load(self.stack_2d)
        self.err = np.load(self.stack_2d_error)
        
        flt = grizli.model.GrismFLT(grism_file= self.grism, 
                                direct_file= self.direct,
                                pad=200, ref_file=None, ref_ext=0, 
                                seg_file='../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.0.F160W_seg.fits',
                                shrink_segimage=False)

        ref_cat = Table.read('../../../Clear_data/goodss_mosaic/goodss_3dhst.v4.3.cat', format='ascii')
        sim_cat = flt.blot_catalog(ref_cat, sextractor=False)

        id = 39170

        x0 = ref_cat['x'][39169]+1
        y0 = ref_cat['y'][39169]+1

        mag =-2.5*np.log10(ref_cat['f_F850LP']) + 25
        keep = mag < 22

        flt.compute_full_model(ids=ref_cat['id'][keep],verbose=False, 
                               mags=mag[keep])

        ### Get the beams/orders
        beam = flt.object_dispersers[id]['A'] # can choose other orders if available
        beam.compute_model()

        ### BeamCutout object
        self.co = grizli.model.BeamCutout(flt, beam, conf=flt.conf)

    def Sim_spec(self, metal, age, tau):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt%s_spec.npy' % (metal, age, tau)
   
        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(self.redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
  
        self.model = self.co.beam.compute_model(spectrum_1d=[spec.wave, spec.flux], 
                                           in_place=False).reshape(self.co.beam.sh_beam)

        adjmodel = np.append(np.zeros([4,len(self.model)]),self.model.T[:-4], axis=0).T
        
        rs = self.gal.shape[0]*self.gal.shape[1]
        C = Scale_model(self.gal.reshape(rs),self.err.reshape(rs),adjmodel.reshape(rs))
    
        self.sim = adjmodel*C
        
def Analyze_2D(chifits, specz, metal, age, tau, age_conv='../data/light_weight_scaling.npy'):
    ####### Get maximum age
    max_age = Oldest_galaxy(specz)

    ####### Read in file
    chi = np.load(chifits).T

    chi[:, len(age[age <= max_age]):, :] = 1E1

    ####### Get scaling factor for tau reshaping
    ultau = np.append(0, np.power(10, np.array(tau)[1:] - 9))

    convtable = np.load(age_conv)

    overhead = np.zeros([len(tau),metal.size]).astype(int)
    for i in range(len(tau)):
        for ii in range(metal.size):
            amt=[]
            for iii in range(age.size):
                if age[iii] > convtable.T[i].T[ii][-1]:
                    amt.append(1)
            overhead[i][ii] = sum(amt)

    ######## Reshape likelihood to get average age instead of age when marginalized
    newchi = np.zeros(chi.shape)

    for i in range(len(chi)):
        frame = np.zeros([metal.size, age.size])
        for ii in range(metal.size):
            dist = interp1d(convtable.T[i].T[ii], chi[i].T[ii])(age[:-overhead[i][ii]])
            frame[ii] = np.append(dist, np.repeat(1E5, overhead[i][ii]))
        newchi[i] = frame.T


    ####### Create normalize probablity marginalized over tau
    P = np.exp(-newchi.T.astype(np.float128) / 2)

    prob = np.trapz(P, ultau, axis=2)
    C = np.trapz(np.trapz(prob, age, axis=1), metal)

    prob /= C

    #### Get Z and t posteriors

    PZ = np.trapz(prob, age, axis=1)
    Pt = np.trapz(prob.T, metal,axis=1)

    return prob.T, PZ,Pt
        
def Single_gal_fit_full_2d(metal, age, tau, specz,stack_2d, stack_2d_error, grism_flt, direct_flt , name):
    #############Read in spectra#################
    spec = Gen_spec_2d(stack_2d, stack_2d_error, grism_flt, direct_flt, specz)

    #############Prep output files: 1-full, 2-cont, 3-feat###############
    chifile1 = '../chidat/%s_chidata' % name

    ##############Create chigrid and add to file#################
    chigrid1 = np.zeros([len(metal),len(age),len(tau)])

    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(tau)):
                spec.Sim_spec(metal[i], age[ii], tau[iii])
                chigrid1[i][ii][iii] = np.sum(((spec.gal - spec.sim)/spec.err)**2)

    ################Write chigrid file###############
    np.save(chifile1,chigrid1)

    print 'Done!'
    return

"""Proposal spec z"""

class Gen_spec_z(object):
    def __init__(self, spec_file, pad=100, minwv = 7900, maxwv = 11400):
        self.galaxy = spec_file
        self.pad = pad

        """ 
        self.flt_input - grism flt (not image flt) which contains the object you're interested in modeling, this
                         will tell Grizli the PA
        **
        self.galaxy_id - used to id galaxy and import spectra
        **
        self.pad - Grizli uses this to add extra pixels to the edge of an image to account for galaxies near the 
                   edge, 100 is usually enough
        **
        self.beam - information used to make models
        **
        self.wv - output wavelength array of simulated spectra
        **
        self.fl - output flux array of simulated spectra
        """


        gal_wv, gal_fl, gal_er = np.load(self.galaxy)
        self.flt_input = '../data/galaxy_flts/n21156_flt.fits'

        IDX = [U for U in range(len(gal_wv)) if minwv <= gal_wv[U] <= maxwv]

        self.gal_wv_rf = gal_wv[IDX] / (1 + 1.251)
        self.gal_wv = gal_wv[IDX]
        self.gal_fl = gal_fl[IDX]
        self.gal_er = gal_er[IDX]

        self.gal_wv_rf = self.gal_wv_rf[self.gal_fl > 0 ]
        self.gal_wv = self.gal_wv[self.gal_fl > 0 ]
        self.gal_er = self.gal_er[self.gal_fl > 0 ]
        self.gal_fl = self.gal_fl[self.gal_fl > 0 ]

        ## Create Grizli model object
        sim_g102 = grizli.model.GrismFLT(grism_file='', verbose=False,
                                         direct_file=self.flt_input,
                                         force_grism='G102', pad=self.pad)

        sim_g102.photutils_detection(detect_thresh=.025, verbose=True, save_detection=True)

        keep = sim_g102.catalog['mag'] < 29
        c = sim_g102.catalog

        sim_g102.compute_full_model(ids=c['id'][keep], mags=c['mag'][keep], verbose=False)

        ## Grab object near the center of the image
        dr = np.sqrt((sim_g102.catalog['x_flt'] - 579) ** 2 + (sim_g102.catalog['y_flt'] - 522) ** 2)
        ix = np.argmin(dr)
        id = sim_g102.catalog['id'][ix]

        ## Spectrum cutouts
        self.beam = grizli.model.BeamCutout(sim_g102, beam=sim_g102.object_dispersers[id]['A'], conf=sim_g102.conf)

    def Sim_spec(self, metal, age, redshift):
        import pysynphot as S
        
        model = '../../../fsps_models_for_fit/fsps_spec/m%s_a%s_dt8.0_spec.npy' % (metal, age)

        wave, fl = np.load(model)
        spec = S.ArraySpectrum(wave, fl, fluxunits='flam')
        spec = spec.redshift(redshift).renorm(1., 'flam', S.ObsBandpass('wfc3,ir,f105w'))
        spec.convert('flam')
        ## Compute the models
        self.beam.compute_model(spectrum_1d=[spec.wave, spec.flux])

        ## Extractions the model (error array here is meaningless)
        w, f, e = self.beam.beam.optimal_extract(self.beam.model, bin=0)

        ifl = interp1d(w, f)(self.gal_wv)

        ## Get sensitivity function
        fwv, ffl = [self.beam.beam.lam, self.beam.beam.sensitivity / np.max(self.beam.beam.sensitivity)]
        filt = interp1d(fwv, ffl)(self.gal_wv)

        adj_ifl = ifl /filt

        C = Scale_model(self.gal_fl, self.gal_er, adj_ifl)

        self.fl = C * adj_ifl


def Specz_fit_2(spec_file, metal, age, rshift, name):
    #############initialize spectra#################
    spec = Gen_spec_z(spec_file)

    #############Prep output file###############
    chifile = '../rshift_dat/%s_z_fit' % name

    ##############Create chigrid and add to file#################
    mfl = np.zeros([len(metal)*len(age)*len(rshift),len(spec.gal_wv)])
    for i in range(len(metal)):
        for ii in range(len(age)):
            for iii in range(len(rshift)):
                spec.Sim_spec(metal[i], age[ii], rshift[iii])
                mfl[i*len(age)*len(rshift)+ii*len(rshift)+iii]=spec.fl
    chigrid = np.sum(((spec.gal_fl - mfl) / spec.gal_er) ** 2, axis=1).reshape([len(metal), len(age), len(rshift)]).\
        astype(np.float128)

    np.save(chifile,chigrid)
    ###############Write chigrid file###############
    Analyze_specz(chifile + '.npy', rshift, metal, age, name)

    print 'Done!'

    return