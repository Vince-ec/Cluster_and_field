from glob import glob
import seaborn as sea
import os
import numpy as np
import matplotlib.pyplot as plt

### grab names
fnm = glob('../data/out_dict/*nestedfit*')

### get object names
nm=[]
for i in fnm:
    nm.append(os.path.basename(i).replace('_nestedfit','').replace('.npy',''))

### extract posteriors

for i in range(len(fnm)):
    results = np.load(fnm[i]).item()
    Z,PZ = sea.distplot(results.samples[:, 0]).get_lines()[0].get_data()
    plt.close()
    np.save('../data/posteriors/{0}_PZ'.format(nm[i]),[Z,PZ / np.trapz(PZ,Z)])
    
    t,Pt = sea.distplot(results.samples[:, 1]).get_lines()[0].get_data()
    plt.close()
    np.save('../data/posteriors/{0}_Pt'.format(nm[i]),[t,Pt / np.trapz(Pt,t)])
    
    tau,Ptau = sea.distplot(results.samples[:, 2]).get_lines()[0].get_data()
    plt.close()
    np.save('../data/posteriors/{0}_Ptau'.format(nm[i]),[tau,Ptau / np.trapz(Ptau,tau)])
    
    z,Pz = sea.distplot(results.samples[:, 3]).get_lines()[0].get_data()
    plt.close()
    np.save('../data/posteriors/{0}_Prs'.format(nm[i]),[z,Pz / np.trapz(Pz,z)])
    
    d,Pd = sea.distplot(results.samples[:, 4]).get_lines()[0].get_data()
    plt.close()
    np.save('../data/posteriors/{0}_Pd'.format(nm[i]),[d,Pd / np.trapz(Pd,d)])
    
