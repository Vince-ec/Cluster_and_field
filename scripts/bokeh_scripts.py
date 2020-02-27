import numpy as np
import pandas as pd
from shutil import copyfile
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
from matplotlib import gridspec
import matplotlib as mpl
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import astropy.units as u

from scipy.interpolate import interp1d, interp2d
from glob import glob
import os


### set home for files
hpath = os.environ['HOME'] + '/'

if hpath == '/Users/Vince.ec/':
    dpath = '/Volumes/Vince_research/Data/' 
    
else:
    dpath = hpath + 'Data/' 

f105N = fits.open('/Volumes/Vince_CLEAR/Data/CLEAR/goodsn-F105W-astrodrizzle-v4.4_drz_sci.fits')
f105N_img = f105N[0].data

f105S = fits.open('/Volumes/Vince_CLEAR/Data/CLEAR/goodss-F105W-astrodrizzle-v4.4_drz_sci.fits')
f105S_img = f105S[0].data

v4N = Table.read(fits.open('/Volumes/Vince_CLEAR/3dhst_V4.4/goodsn_3dhst.v4.4.cats/Eazy/goodsn_3dhst.v4.4.zout.fits'),
                 format='fits').to_pandas()
v4S = Table.read(fits.open('/Volumes/Vince_CLEAR/3dhst_V4.4/goodss_3dhst.v4.4.cats/Eazy/goodss_3dhst.v4.4.zout.fits'),
                 format='fits').to_pandas()

wfN = wcs.WCS(f105N[0].header)
wfS = wcs.WCS(f105S[0].header)

def RS_img(img):
    IMG = np.array(img) + 100

    m = np.percentile(IMG, 5)
    M = np.percentile(IMG, 99)

    IMG -= m
    IMG[IMG <= 0] = 0
    IMG /= (M-m)
    IMG[IMG > 1] =1
    return np.arcsinh(IMG)

def get_positions(ra, dec, D, W):
    [Of, Lf, Hf]=W.wcs_world2pix([[ra ,dec],[ra-D ,dec-D], [ra+D ,dec+D]],1)
    Of = Of.astype(int)
    Lf = Lf.astype(int)
    Hf = Hf.astype(int)
    return Of, Lf, Hf

def img_ext(field, gid):
    D = (1.5 * u.arcsec * (1*u.arcmin/(60*u.arcsec)) * (1*u.deg/(60*u.arcmin))).value

    if field == 'GND':
        ra=v4N.query('id == {}'.format(gid)).ra.values[0]
        dec=v4N.query('id == {}'.format(gid)).dec.values[0]
        W = wfN
    if field == 'GSD':
        ra=v4S.query('id == {}'.format(gid)).ra.values[0]
        dec=v4S.query('id == {}'.format(gid)).dec.values[0]
        W = wfS

    Of1, Lf1, Hf1 = get_positions(ra, dec, D, W)

    dx = np.abs(Of1[0] - Lf1[0])

    if field == 'GND':
        gal_img = f105N_img[Of1[1] - dx : Of1[1]+1 + dx , Of1[0] - dx: Of1[0]+1 + dx]
    if field == 'GSD':
        gal_img = f105S_img[Of1[1] - dx : Of1[1]+1 + dx , Of1[0] - dx: Of1[0]+1 + dx]
    return gal_img