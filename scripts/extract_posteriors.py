from glob import glob
import seaborn as sea
import os
import numpy as np
import matplotlib.pyplot as plt
from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde

def Get_posterior(sample,logwt,logz):
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

### grab names
fnm = glob('../data/out_dict/G*nestedfit*')

### get object names
nm=[]
for i in fnm:
    nm.append(os.path.basename(i).replace('_nestedfit','').replace('.npy',''))

### extract posteriors

for i in range(len(fnm)):
    results = np.load(fnm[i]).item()
    
    Z,PZ = Get_posterior(results.samples[:, 0], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_PZ'.format(nm[i]),[Z,PZ / np.trapz(PZ,Z)])
    
    t,Pt = Get_posterior(results.samples[:, 1], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Pt'.format(nm[i]),[t,Pt / np.trapz(Pt,t)])
    
    tau,Ptau = Get_posterior(results.samples[:, 2], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau1'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    tau,Ptau = Get_posterior(results.samples[:, 3], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau2'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    tau,Ptau = Get_posterior(results.samples[:, 4], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau3'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    tau,Ptau = Get_posterior(results.samples[:, 5], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau4'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    tau,Ptau = Get_posterior(results.samples[:, 6], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau5'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    tau,Ptau = Get_posterior(results.samples[:, 7], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Ptau6'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    z,Pz = Get_posterior(results.samples[:, 8], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Prs'.format(nm[i]),[z,Pz / np.trapz(Pz,z)])
    
    d,Pd = Get_posterior(results.samples[:, 9], results.logwt,results.logz)
    np.save('../data/posteriors/{0}_Pd'.format(nm[i]),[d,Pd / np.trapz(Pd,d)])
    
