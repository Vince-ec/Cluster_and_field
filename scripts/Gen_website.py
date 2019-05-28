import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd
import os
from astropy.io import fits
from astropy.table import Table
import img_scale
from astropy.wcs import wcs
from glob import glob
import warnings
from astropy.modeling import models, fitting


sea.set(style='white')
sea.set(style='ticks')
sea.set_style({'xtick.direct'
               'ion': 'in','xtick.top':True,'xtick.minor.visible': True,
               'ytick.direction': "in",'ytick.right': True,'ytick.minor.visible': True})
cmap = sea.cubehelix_palette(12, start=2, rot=.2, dark=0, light=1.0, as_cmap=True) 

### set home for files
hpath = os.environ['HOME'] + '/'

gsd_cat = Table.read('/Volumes/Vince_CLEAR/3dhst_V4.4/goodss_3dhst.v4.4.cats/Catalog/goodss_3dhst.v4.4.cat', format='ascii').to_pandas()
gnd_cat = Table.read('/Volumes/Vince_CLEAR/3dhst_V4.4/goodsn_3dhst.v4.4.cats/Catalog/goodsn_3dhst.v4.4.cat', format='ascii').to_pandas()

field_dir = 's'

seg = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F160W_seg.fits'.format(field_dir))
f160 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F160W_orig_sci.fits'.format(field_dir))
f125 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F125W_orig_sci.fits'.format(field_dir))
f105 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}-F105W-astrodrizzle-v4.4_drz_sci.fits'.format(field_dir))

fields = ['GS1', 'GS2', 'GS3', 'GS4', 'GS5', 'ERSPRIME']

for field in fields:
    G_flist = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/{0}/*1D.fits'.format(field))

    G_ids = []

    for i in range(len(G_flist)):
        glist = fits.open(G_flist[i])[0].header['GRIS*']
        if 'G102' in [glist[U] for U in range(len(glist))]:
            G_ids.append(int(os.path.basename(G_flist[i]).split('.')[0].split('_')[1]))

    cat = gsd_cat

    ra,dec = [[],[]]
    kr = []
    for i in G_ids:
        ra.append(cat.query('id == {0}'.format(i)).ra.values[0] )
        dec.append(cat.query('id == {0}'.format(i)).dec.values[0] )    
        kr.append(cat.query('id == {0}'.format(i)).kron_radius.values[0])

    G_DF = pd.DataFrame({'ids':G_ids, 'ra' : ra, 'dec' : dec, 'kr':kr})

    w = wcs.WCS(f125[0].header)

    pos = w.wcs_world2pix(np.array([G_DF.ra,G_DF.dec]).T, 1)

    G_DF['x'] = pos.T[0]
    G_DF['y'] = pos.T[1]

    xlims = np.array([min(pos.T[0]) - 20, max(pos.T[0]) + 20]).astype(int)
    ylims = np.array([min(pos.T[1])- 20, max(pos.T[1])+ 20]).astype(int) 

    f105img = f105[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]
    f125img = f125[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]
    f160img = f160[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]

    img = np.zeros((f125img.shape[0], f125img.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.asinh(f160img, scale_min=-0.1, scale_max=0.5)
    img[:,:,1] = img_scale.asinh(f125img, scale_min=-0.1, scale_max=0.5)
    img[:,:,2] = img_scale.asinh(f105img, scale_min=-0.1, scale_max=0.5)

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(img.shape[1] / float(DPI), img.shape[0] / float(DPI))

    plt.imshow(img,aspect='equal')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../data/website_data/{0}_img.pdf'.format(field))#, bbox_inches = 'tight')

    xarray = np.arange(1,img.shape[1] + 1)
    yarray = np.arange(1,img.shape[0] + 1)

    xpos = w.wcs_pix2world(np.array([xarray, np.zeros_like(xarray)]).T, 1)
    ypos = w.wcs_pix2world(np.array([np.zeros_like(yarray), yarray]).T, 1)

    dra = (xpos.T[0][0] - xpos.T[0][-1]) / img.shape[1]
    ddec = - (ypos.T[1][0] - ypos.T[1][-1]) / img.shape[0]

    G_DF['x'] = pos.T[0] - xlims[0]
    G_DF['y'] = pos.T[1] - ylims[0]
    G_DF = G_DF[['ids', 'ra', 'dec', 'x', 'y', 'kr']]
    G_DF.to_csv('../data/website_data/{0}.cat'.format(field), index = False, sep = ' ')

    xrange = np.arange(xlims[0], xlims[1] + 1,1)
    yrange = np.arange(ylims[0], ylims[1] + 1,1)
    RA = np.zeros([len(xrange),len(yrange)])
    DEC = np.zeros([len(xrange),len(yrange)])
    for i in range(len(xrange)):
        for ii in range(len(yrange)):
            RA[i][ii],DEC[i][ii] = w.wcs_pix2world(np.array([[xrange[i],yrange[ii]]]), 1)[0]

    xrnorm = np.arange(len(xrange))
    yrnorm = np.arange(len(yrange))

    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    ygrid, xgrid = np.meshgrid(yrnorm,xrnorm) 
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p_ra = fit_p(p_init, xgrid, ygrid, RA)

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p_dec = fit_p(p_init, xgrid, ygrid, DEC)

    pr_p = p_ra.param_sets.T[0]
    pd_p = p_dec.param_sets.T[0]

    np.savetxt('../data/website_data/{0}_RA_posistion.dat'.format(field), pr_p,fmt ='%s')
    np.savetxt('../data/website_data/{0}_DEC_posistion.dat'.format(field), pd_p,fmt ='%s')


field_dir = 'n'

seg = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F160W_seg.fits'.format(field_dir))
f160 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F160W_orig_sci.fits'.format(field_dir))
f125 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}_3dhst.v4.0.F125W_orig_sci.fits'.format(field_dir))
f105 = fits.open('/Volumes/Vince_CLEAR/g{0}d_img/goods{0}-F105W-astrodrizzle-v4.4_drz_sci.fits'.format(field_dir))

fields = ['GN1', 'GN2', 'GN3', 'GN4', 'GN5', 'GN7']

for field in fields:
    G_flist = glob('/Volumes/Vince_CLEAR/RELEASE_v2.1.0/{0}/*1D.fits'.format(field))

    G_ids = []

    for i in range(len(G_flist)):
        glist = fits.open(G_flist[i])[0].header['GRIS*']
        if 'G102' in [glist[U] for U in range(len(glist))]:
            G_ids.append(int(os.path.basename(G_flist[i]).split('.')[0].split('_')[1]))

    cat = gnd_cat

    ra,dec = [[],[]]
    kr = []
    for i in G_ids:
        ra.append(cat.query('id == {0}'.format(i)).ra.values[0] )
        dec.append(cat.query('id == {0}'.format(i)).dec.values[0] )    
        kr.append(cat.query('id == {0}'.format(i)).kron_radius.values[0])

    G_DF = pd.DataFrame({'ids':G_ids, 'ra' : ra, 'dec' : dec, 'kr':kr})

    w = wcs.WCS(f125[0].header)

    pos = w.wcs_world2pix(np.array([G_DF.ra,G_DF.dec]).T, 1)

    G_DF['x'] = pos.T[0]
    G_DF['y'] = pos.T[1]

    xlims = np.array([min(pos.T[0]) - 20, max(pos.T[0]) + 20]).astype(int)
    ylims = np.array([min(pos.T[1])- 20, max(pos.T[1])+ 20]).astype(int) 

    f105img = f105[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]
    f125img = f125[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]
    f160img = f160[0].data[ylims[0]:ylims[1],xlims[0]:xlims[1]]

    img = np.zeros((f125img.shape[0], f125img.shape[1], 3), dtype=float)
    img[:,:,0] = img_scale.asinh(f160img, scale_min=-0.1, scale_max=0.5)
    img[:,:,1] = img_scale.asinh(f125img, scale_min=-0.1, scale_max=0.5)
    img[:,:,2] = img_scale.asinh(f105img, scale_min=-0.1, scale_max=0.5)

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(img.shape[1] / float(DPI), img.shape[0] / float(DPI))

    plt.imshow(img,aspect='equal')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('../data/website_data/{0}_img.pdf'.format(field))#, bbox_inches = 'tight')

    xarray = np.arange(1,img.shape[1] + 1)
    yarray = np.arange(1,img.shape[0] + 1)

    xpos = w.wcs_pix2world(np.array([xarray, np.zeros_like(xarray)]).T, 1)
    ypos = w.wcs_pix2world(np.array([np.zeros_like(yarray), yarray]).T, 1)

    dra = (xpos.T[0][0] - xpos.T[0][-1]) / img.shape[1]
    ddec = - (ypos.T[1][0] - ypos.T[1][-1]) / img.shape[0]

    G_DF['x'] = pos.T[0] - xlims[0]
    G_DF['y'] = pos.T[1] - ylims[0]
    G_DF = G_DF[['ids', 'ra', 'dec', 'x', 'y', 'kr']]
    G_DF.to_csv('../data/website_data/{0}.cat'.format(field), index = False, sep = ' ')

    xrange = np.arange(xlims[0], xlims[1] + 1,1)
    yrange = np.arange(ylims[0], ylims[1] + 1,1)
    RA = np.zeros([len(xrange),len(yrange)])
    DEC = np.zeros([len(xrange),len(yrange)])
    for i in range(len(xrange)):
        for ii in range(len(yrange)):
            RA[i][ii],DEC[i][ii] = w.wcs_pix2world(np.array([[xrange[i],yrange[ii]]]), 1)[0]

    xrnorm = np.arange(len(xrange))
    yrnorm = np.arange(len(yrange))

    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    ygrid, xgrid = np.meshgrid(yrnorm,xrnorm) 
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p_ra = fit_p(p_init, xgrid, ygrid, RA)

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p_dec = fit_p(p_init, xgrid, ygrid, DEC)

    pr_p = p_ra.param_sets.T[0]
    pd_p = p_dec.param_sets.T[0]

    np.savetxt('../data/website_data/{0}_RA_posistion.dat'.format(field), pr_p,fmt ='%s')
    np.savetxt('../data/website_data/{0}_DEC_posistion.dat'.format(field), pd_p,fmt ='%s')
