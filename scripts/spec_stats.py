__author__ = 'vestrada'

import numpy as np
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
R = robjects.r
pandas2ri.activate()


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

def Best_fit_model(input_file, metal, age, tau):
    dat = fits.open(input_file)

    chi = []
    for i in range(len(metal)):
        chi.append(dat[i + 1].data)
    chi = np.array(chi)

    x = np.argwhere(chi == np.min(chi))
    print( metal[x[0][0]], age[x[0][1]], tau[x[0][2]])
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

def Iterative_stacking(grid_o,x_o, extend=False, iterations = 20,resampling = 250):
    ksmooth = importr('KernSmooth')
    del_x = x_o[1] - x_o[0]
    rto = int(np.abs(min(np.log10(x_o)[np.abs(np.log10(x_o)) != np.inf])))+1

    if extend:
        x_n,grid_n = Reconfigure_dist(grid_o,x_o,rto)

        x = np.linspace(x_n[0],x_n[-1],resampling)
        grid = np.zeros([len(grid_n),x.size])    
        for i in range(len(grid_n)):
            grid[i] = interp1d(x_n,grid_n[i])(x)
        ### select bandwidth
        H = ksmooth.dpik(x_o) 
    else:
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
    rsFx = interp1d(x,Fx)(x_o)
    rsFx = rsFx/np.trapz(rsFx,x_o)  
    
    rsstkpos = interp1d(x,stkpos)(x_o)
    rsstkpos = rsstkpos/np.trapz(rsstkpos,x_o)  
    return rsFx,rsstkpos

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

def Highest_density_region(Px, x, region = 0.683):
    mode = x[Px == max(Px)][0]

    pbin = np.linspace(0, max(Px), 1000)
    pbin = pbin[::-1]

    prob_int = np.zeros(len(pbin))

    for i in range(len(pbin)):
        p = np.array(Px)
        p[p <= pbin[i]] = 0
        prob_int[i] = np.trapz(p, x)
        if prob_int[i] > region:
            break

    HCI = []
    flip = False
    for i in range(len(x)):
        if flip == False:
            if p[i] != 0:
                HCI.append(x[i])
                flip = True
        else:
            if p[i] == 0:
                HCI.append(x[i-1])
                flip = False
            if x[i] == x[-1]:
                HCI.append(x[i])    
                                   
    off_main_mass = []
            
    for i in range(len(HCI) // 2):
        IDX = [U for U in range(len(x)) if HCI[2*i] <= x[U] <= HCI[2*i+1]]
        
        if HCI[2*i] <= mode <= HCI[2*i+1]:
            mode_reg = [HCI[2*i],HCI[2*i+1]]
            main_mass = np.trapz(Px[IDX], x[IDX])
        
        else:
            off_main_mass.append(np.trapz(Px[IDX], x[IDX]))
        
    if len(HCI) // 2 == 1:
        off_main_mass.append(0)
        
    return mode, mode_reg, np.array(off_main_mass) / main_mass

def Smooth(f,x,bw):
    ksmooth = importr('KernSmooth')

    ### select bandwidth
    H = ksmooth.dpik(x)
    
    if bw == 'none':
        bw = H
    
    fx = ksmooth.locpoly(x,f,bandwidth = bw)
    X = np.array(fx[0])
    iFX = np.array(fx[1])
    return interp1d(X,iFX)(x)

def Get_posterior(results, entry):
    sample = results.samples[:, entry]
    logwt = results.logwt
    logz = results.logz
    
    
    weight = np.exp(logwt - logz[-1])

    q = [0.5 - 0.5 * 0.999999426697, 0.5 + 0.5 * 0.999999426697]
    span = _quantile(sample.T, q, weights=weight)

    s = 0.02

    bins = int(round(10. / 0.02))
    n, b = np.histogram(sample, bins=bins, weights=weight,
                        range=np.sort(span))
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    
    return x0, y0 / np.trapz(y0,x0)

def Get_derived_posterior(sample, results):
    logwt = results.logwt
    logz = results.logz
    
    
    weight = np.exp(logwt - logz[-1])

    q = [0.5 - 0.5 * 0.999999426697, 0.5 + 0.5 * 0.999999426697]
    span = _quantile(sample.T, q, weights=weight)

    s = 0.02

    bins = int(round(10. / 0.02))
    n, b = np.histogram(sample, bins=bins, weights=weight,
                        range=np.sort(span))
    n = norm_kde(n, 10.)
    x0 = 0.5 * (b[1:] + b[:-1])
    y0 = n
    
    return x0, y0 / np.trapz(y0,x0)