__author__ = 'vestrada'

import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy import wcs
import os
### set home for files
hpath = os.environ['HOME'] + '/'

h=6.6260755E-27 # planck constant erg s
c=3E10          # speed of light cm s^-1
atocm=1E-8    # unit to convert angstrom to cm
kb=1.38E-16	    # erg k-1

def Mag(band):
    magnitude=25-2.5*np.log10(band)
    return magnitude

def Oldest_galaxy(z):
    return cosmo.age(z).value

def Gauss_dist(x, mu, sigma):
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return G / np.trapz(G, x)

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

def Source_present(fn,ra,dec):  ### finds source in flt file, returns if present and the pos in pixels
    flt=fits.open(fn)
    present = False
    
    w = wcs.WCS(flt[1].header)

    xpixlim=len(flt[1].data[0])
    ypixlim=len(flt[1].data)

    [pos]=w.wcs_world2pix([[ra,dec]],1)

    if -100 <pos[0]< (xpixlim - 35) and 0 <pos[1]<ypixlim and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
        present=True
            
    return present,pos
    
def Get_Sensitivity(filter_num):
    f=open(hpath + 'GitHub/Quiescent_analysis/scripts/vtl/FILTER.RES.latest','r')
    data=f.readlines()
    rows=[]
    for i in range(len(data)):
        rows.append(data[i].split())

    i=0
    sens_data=[]
    while i < len(data):
        sdata=[]
        amount=int(rows[i][0])
        for u in range(amount):
            r=np.array(rows[i+u+1])
            sdata.append(r.astype(np.float))
        sens_data.append(sdata)
        i=i+amount+1

    sens_wave=[]
    sens_func=[]
    s_wave=[]
    s_func=[]
    for i in range(len(sens_data[filter_num-1])):
        s_wave.append(sens_data[filter_num-1][i][1])
        s_func.append(sens_data[filter_num-1][i][2])

    for i in range(len(s_func)):
        if .001 < s_func[i]:
            sens_func.append(s_func[i])
            sens_wave.append(s_wave[i])

    
    trans = np.array(sens_func) / np.max(sens_func)
    sens_wv = np.array(sens_wave)
    
    tp = np.trapz(((trans * np.log(sens_wv)) / sens_wv), sens_wv)
    bm = np.trapz(trans / sens_wv, sens_wv)

    wave_eff = np.exp(tp / bm)
    
    return sens_wv, trans, wave_eff

def Sig_int(nu,er,trans,energy):
    sig = np.zeros(len(nu)-1)
    
    for i in range(len(nu)-1):
        sig[i] = (nu[i+1] - nu[i])/2 *np.sqrt(er[i]**2 * energy[i]**2 * trans[i]**2 + er[i+1]**2 * energy[i+1]**2 * trans[i+1]**2)
    
    return np.sum(sig) / np.trapz(trans * energy, nu)

class Photometry(object):

    def __init__(self,wv,fl,er,filter_number):
        self.wv = wv
        self.fl = fl
        self.er = er
        self.filter_number = filter_number

    def Get_Sensitivity(self, filter_num = 0):
        if filter_num != 0:
            self.filter_number = filter_num

        f = open('~/GitHub/Quiescent_analysis/scripts/vtl/FILTER.RES.latest', 'r')
        data = f.readlines()
        rows = []
        for i in range(len(data)):
            rows.append(data[i].split())
        i = 0
        sens_data = []
        while i < len(data):
            sdata = []
            amount = int(rows[i][0])
            for u in range(amount):
                r = np.array(rows[i + u + 1])
                sdata.append(r.astype(np.float))
            sens_data.append(sdata)
            i = i + amount + 1

        sens_wave = []
        sens_func = []
        s_wave = []
        s_func = []
        for i in range(len(sens_data[self.filter_number - 1])):
            s_wave.append(sens_data[self.filter_number - 1][i][1])
            s_func.append(sens_data[self.filter_number - 1][i][2])

        for i in range(len(s_func)):
            if .001 < s_func[i]:
                sens_func.append(s_func[i])
                sens_wave.append(s_wave[i])

        self.sens_wv = np.array(sens_wave)
        self.trans = np.array(sens_func)

    def Photo(self):
        wave = self.wv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans,energy) * (c / (wave_eff * atocm) ** 2)

    def Photo_clipped(self):

        IDX = [U for U in range(len(self.sens_wv)) if self.wv[0] < self.sens_wv[U] < self.wv[-1]]

        wave = self.wv * atocm
        filtnu = c /(self.sens_wv[IDX] * atocm)
        nu = c / wave
        fnu = (c/nu**2) * self.fl
        Fnu = interp1d(nu, fnu)(filtnu)
        ernu = (c/nu**2) * self.er
        Ernu = interp1d(nu, ernu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans[IDX]
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans[IDX] * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        top1 = Ernu * energy * self.trans[IDX]
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans[IDX] * energy
        bottom = np.trapz(bottom1, filtnu)
        erphotonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)

        photo = photonu * (c / (wave_eff * atocm) ** 2)
        photoer = erphotonu * (c / (wave_eff * atocm) ** 2)

        self.eff_wv = wave_eff
        self.photo = photo
        self.photo_er = Sig_int(filtnu,Ernu,self.trans[IDX],energy) * (c / (wave_eff * atocm) ** 2)
        
    def Photo_model(self,mwv,mfl):

        wave = mwv * atocm
        filtnu = c /(self.sens_wv * atocm)
        nu = c / wave
        fnu = (c/nu**2) * mfl
        Fnu = interp1d(nu, fnu)(filtnu)

        energy = 1 / (h *filtnu)

        top1 = Fnu * energy * self.trans
        top = np.trapz(top1, filtnu)
        bottom1 = self.trans * energy
        bottom = np.trapz(bottom1, filtnu)
        photonu = top / bottom

        tp = np.trapz(((self.trans * np.log(self.sens_wv)) / self.sens_wv), self.sens_wv)
        bm = np.trapz(self.trans / self.sens_wv, self.sens_wv)

        wave_eff = np.exp(tp / bm)
        photo = photonu * (c / (wave_eff * atocm) ** 2)
        
        self.eff_mwv = wave_eff
        self.mphoto = photo

    def FWHM(self):
        top = np.trapz((self.trans * np.log(self.sens_wv/self.eff_wv)**2) / self.sens_wv, self.sens_wv)
        bot = np.trapz(self.trans / self.sens_wv, self.sens_wv)
        sigma = np.sqrt(top/bot)

        self.fwhm = np.sqrt(8*np.log(2))*sigma * self.eff_wv
