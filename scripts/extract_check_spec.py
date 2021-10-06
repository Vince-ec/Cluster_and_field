import numpy as np
from spec_id import *
import grizli
import fsps

# NGSID = np.load(data_path + 'N_GSD.npy', allow_pickle=True)
# NGNID = np.load(data_path + 'N_GND.npy', allow_pickle=True)

# NGSz = np.load(data_path + 'N_GSD_z.npy', allow_pickle=True)
# NGNz = np.load(data_path + 'N_GND_z.npy', allow_pickle=True)

NGSID = np.load('../dataframes/N_GSD.npy', allow_pickle=True)
NGNID = np.load('../dataframes/N_GND.npy', allow_pickle=True)

NGSz = np.load('../dataframes/N_GSD_z.npy', allow_pickle=True)
NGNz = np.load('../dataframes/N_GND_z.npy', allow_pickle=True)

class Gen_spec_check(object):
    def __init__(self, field, galaxy_id, specz,
                 g102_lims=[8000, 11300], g141_lims=[11300, 16500]):
        self.field = field
        self.galaxy_id = galaxy_id
        self.specz = specz
        self.c = 3E18          # speed of light angstrom s^-1
        self.g102_lims = g102_lims
        self.g141_lims = g141_lims

        """
        B - prefix refers to g102
        R - prefix refers to g141
        P - prefix refers to photometry
        
        field - GND/GSD
        galaxy_id - ID number from 3D-HST
        specz - redshift

        """
        ##load spec and phot
        self.make_multibeam()
            
        if self.g102:
            self.Bwv, self.Bfl, self.Ber = self.Gen_1D_spec(self.mb_g102, g102_lims, 'G102', self.specz,)
            self.Bwv_rf = self.Bwv/(1 + self.specz)

        if self.g141:
            self.Rwv, self.Rfl, self.Rer = self.Gen_1D_spec(self.mb_g141, g141_lims, 'G141', self.specz,)
            self.Rwv_rf = self.Rwv/(1 + self.specz)
        
     
    def make_multibeam(self):
        if int(self.galaxy_id) < 10000:
            gid = '0' + str(self.galaxy_id)
        else:
            gid = self.galaxy_id

        if hpath == '/home/vestrada78840/':
            fl = glob(beam_2d_path + '*{}*{}*'.format(self.field[1], gid))
        else:
            fl = glob(beam_2d_path + '*{}*/*{}*'.format(self.field[1], gid))
        
        sz = []
        for f in fl:
            sz.append(os.path.getsize(f))
       
        fl = np.array(fl)[np.argsort(sz)]

        nlist = []
        blist = []
        for f in fl:
            mb = multifit.MultiBeam(f,**args)
            for bm in mb.beams:
                if bm.grism.parent_file not in nlist:
                    nlist.append(bm.grism.parent_file)
                    blist.append(bm)


        mb = multifit.MultiBeam(blist,**args)
        for b in mb.beams:
            if hasattr(b, 'xp'):
                delattr(b, 'xp')
        mb.initialize_masked_arrays()

        grism_beams = {}
        for g in mb.PA:
            grism_beams[g.lower()] = []
            for pa in mb.PA[g]:
                for i in mb.PA[g][pa]:
                    grism_beams[g.lower()].append(mb.beams[i])

        try:
            self.mb_g102 = multifit.MultiBeam(grism_beams['g102'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g102')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g102.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g102.initialize_masked_arrays()
            self.g102 = True
            
        except:
            self.g102 = False
            
        try:
            self.mb_g141 = multifit.MultiBeam(grism_beams['g141'], fcontam=mb.fcontam, 
                                         min_sens=mb.min_sens, min_mask=mb.min_mask, 
                                         group_name=mb.group_name+'-g141')
            # bug, will be fixed ~today to not have to do this in the future
            for b in self.mb_g141.beams:
                if hasattr(b, 'xp'):
                    delattr(b, 'xp')
            self.mb_g141.initialize_masked_arrays()
            self.g141 = True
            
        except:
            self.g141 = False
            
    def Gen_1D_spec(self, MB, lims, instr, specz):
        temps = MB.template_at_z(specz, templates = args['t1'], fitter='lstsq')
        sptbl = MB.oned_spectrum(tfit = temps)

        w = sptbl[instr]['wave']
        f = sptbl[instr]['flux']
        e = sptbl[instr]['err']
        fl = sptbl[instr]['flat']

        clip = [U for U in range(len(w)) if lims[0] < w[U] < lims[1]]
        
        w = w[clip]
        f = f[clip]
        e = e[clip]
        fl = fl[clip]

        return w[f>0], f[f>0]/fl[f>0], e[f>0]/fl[f>0]
    
# gsd_spec ={}
gnd_spec ={}
  
# for i in range(len(NGSID)):
#     GS = Gen_spec_check('GSD', NGSID[i],NGSz[i])
#     gsd_spec['{}'.format(NGSID[i])] = {}
#     if GS.g102:
#         gsd_spec['{}'.format(NGSID[i])]['g102_wv']=GS.Bwv
#         gsd_spec['{}'.format(NGSID[i])]['g102_fl']=GS.Bfl
#         gsd_spec['{}'.format(NGSID[i])]['g102_er']=GS.Ber
            
#     if GS.g141:    
#         gsd_spec['{}'.format(NGSID[i])]['g141_wv']=GS.Rwv
#         gsd_spec['{}'.format(NGSID[i])]['g141_fl']=GS.Rfl
#         gsd_spec['{}'.format(NGSID[i])]['g141_er']=GS.Rer
        
for i in range(len(NGNID)):
    GS = Gen_spec_check('GND', NGNID[i],NGNz[i])
    gnd_spec['{}'.format(NGNID[i])] = {}
    
    if GS.g102:
        gnd_spec['{}'.format(NGNID[i])]['g102_wv']=GS.Bwv
        gnd_spec['{}'.format(NGNID[i])]['g102_fl']=GS.Bfl
        gnd_spec['{}'.format(NGNID[i])]['g102_er']=GS.Ber
            
    if GS.g141:    
        gnd_spec['{}'.format(NGNID[i])]['g141_wv']=GS.Rwv
        gnd_spec['{}'.format(NGNID[i])]['g141_fl']=GS.Rfl
        gnd_spec['{}'.format(NGNID[i])]['g141_er']=GS.Rer
  
# np.save(data_path + 'GSD_new', gsd_spec, allow_pickle = True)
np.save(data_path + 'GND_new', gnd_spec, allow_pickle = True)
    
gsd_spec = np.load(data_path + 'GSD_new.npy', allow_pickle = True).item()
    
for i in range(len(NGSID)):
    plt.figure(figsize = [12,6])
    
    try:
        Bwv = gsd_spec['{}'.format(NGSID[i])]['g102_wv']
        Bfl = gsd_spec['{}'.format(NGSID[i])]['g102_fl']
        Ber = gsd_spec['{}'.format(NGSID[i])]['g102_er']
        plt.errorbar(Bwv,Bfl,Ber, fmt = 'o', color ='b')
    except:
        pass
    try:
        Rwv = gsd_spec['{}'.format(NGSID[i])]['g141_wv']
        Rfl = gsd_spec['{}'.format(NGSID[i])]['g141_fl']
        Rer = gsd_spec['{}'.format(NGSID[i])]['g141_er']
        plt.errorbar(Rwv,Rfl,Rer, fmt = 'o', color ='r')
    except:
        pass
    plt.title('GSD-{}, z={}'.format(NGSID[i], NGSz[i]))
    plt.savefig('../plots/newspec_check/GSD_{}.png'.format(NGSID[i]), bbox_inches = 'tight')
        
        
for i in range(len(NGNID)):
    plt.figure(figsize = [12,6])
    
    try:
        Bwv = gnd_spec['{}'.format(NGNID[i])]['g102_wv']
        Bfl = gnd_spec['{}'.format(NGNID[i])]['g102_fl']
        Ber = gnd_spec['{}'.format(NGNID[i])]['g102_er']
        plt.errorbar(Bwv,Bfl,Ber, fmt = 'o', color ='b')
    except:
        pass
    try:
        Rwv = gnd_spec['{}'.format(NGNID[i])]['g141_wv']
        Rfl = gnd_spec['{}'.format(NGNID[i])]['g141_fl']
        Rer = gnd_spec['{}'.format(NGNID[i])]['g141_er']
        plt.errorbar(Rwv,Rfl,Rer, fmt = 'o', color ='r')
    except:
        pass
    plt.title('GND-{}, z={}'.format(NGNID[i], NGNz[i]))
    plt.savefig('../plots/newspec_check/GND_{}.png'.format(NGNID[i]), bbox_inches = 'tight')