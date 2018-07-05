__author__ = 'vestrada'

import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import Planck13 as cosmo
from astropy.io import fits
from astropy import wcs

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
    pos = [0,0]
    
    if np.abs(dec - flt[0].header['DEC_TARG']) < 1:
        w = wcs.WCS(flt[1].header)

        xpixlim=len(flt[1].data[0])
        ypixlim=len(flt[1].data)

        [pos]=w.wcs_world2pix([[ra,dec]],1)

        if -100<pos[0]<xpixlim + 100 and -100<pos[1]<ypixlim + 100 and flt[0].header['OBSTYPE'] == 'SPECTROSCOPIC':
            present=True
            
    return present,pos
    

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