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

"""Single Galaxy"""
 
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




"""Proposal fit 2D"""
        
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