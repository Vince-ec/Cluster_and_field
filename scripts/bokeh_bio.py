import pandas as pd
import numpy as np
from shutil import copy
from glob import glob
from bokeh.models import HoverTool, ColumnDataSource, OpenURL, TapTool, DataTable, TableColumn, Label, BoxAnnotation
from bokeh import palettes
from bokeh.plotting import figure,save
from bokeh.io import show, output_notebook, output_file
from bokeh.transform import linear_cmap
from bokeh.layouts import gridplot

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
from astropy.cosmology import z_at_value
import astropy.units as u

class I_catalog(object):
    def __init__(self):
        
        alldb = pd.read_pickle('../dataframes/fitdb/allfits_1D.pkl')
        morph_db = alldb.query('W_UVJ == "Q" and AGN != "AGN" and lmass >= 10.5 and n_f < 3 and Re < 20 ')

        bspec = [27458,294464,36348,48631,19290,32566,32691,33093,26272,35640,45333, 30144, 21683]
        nog102 = [27714,37189,26139,32799,47223,22774,28890,23073,31452,24033]

        inout = []
        for i in morph_db.index:     
            if morph_db.id[i] not in bspec and morph_db.id[i] not in nog102: 
                inout.append('i')
            else:
                inout.append('o')

        morph_db['inout'] = inout
        self.DB = morph_db.query('inout == "i" and 0.7 < zgrism < 2.5 and Sigma1 > 10**9.6')
        
        self.g102 = pd.read_pickle('../bokeh_app/data/G102.pkl')
        self.g141 = pd.read_pickle('../bokeh_app/data/G141.pkl')
        self.phot = pd.read_pickle('../bokeh_app/data/PHOT.pkl')
        self.model = pd.read_pickle('../bokeh_app/data/MODEL.pkl')
        self.SFH = pd.read_pickle('../bokeh_app/data/SFH.pkl')
        
        self.s1 = ColumnDataSource(data = {
            'id':self.DB.id,'S1':np.log10(self.DB.Sigma1), 
            'z50':cosmo.lookback_time(self.DB.z_50),
            'z50p':self.DB.z_50, 't50':self.DB.t_50, 'sSFR':self.DB.log_ssfr, 
            'zgrism':cosmo.lookback_time(self.DB.zgrism),
            'zgrismp':self.DB.zgrism, 'Re':self.DB.Re, 
            'lmass':self.DB.lmass, 'tq':self.DB.t_50-self.DB.t_90})

        self.s2 = ColumnDataSource(data = {
            'id':self.DB.id,'S1':np.log10(self.DB.Sigma1), 
            'z50':self.DB.z_50, 't50':self.DB.t_50, 'sSFR':self.DB.log_ssfr, 
            'zgrism':self.DB.zgrism,
            'Re':self.DB.Re, 
            'lmass':self.DB.lmass, 'tq':self.DB.t_50-self.DB.t_90})
    
zs = cosmo.lookback_time([2,3,4,5,6,7]).value
zg = cosmo.lookback_time([0.7, 1.0, 1.3, .16, 1.9, 2.2]).value
cats = I_catalog()

def IMG_plot(gid):    
    IMG = np.load('../bokeh_app/data/imgs/{}_img.npy'.format(gid))
    img = figure(plot_width=300, plot_height=225, x_range=(0, 10), y_range=(0, 10), tools = '')
    img.image(image=[IMG], x=0, y=0, dw=10, dh=10,palette=palettes.gray(100))
    img.title.text_font_size = "20pt"
    img.xaxis.major_label_text_color = img.yaxis.major_label_text_color = None
    img.yaxis.major_tick_line_color = img.xaxis.major_tick_line_color =None 
    img.yaxis.minor_tick_line_color = img.xaxis.minor_tick_line_color =None   
    return img
    
def SFH_plot(DB, SDB, gid):
    rshift = DB.query('id == {}'.format(gid)).zgrism.values[0]
    LBT = SDB.LBT
    SFH = SDB['{}'.format(gid)]
    LBT = LBT[SFH**2 > 0]
    SFH = SFH[SFH**2 > 0]
    
    zs = [z_at_value(cosmo.lookback_time,(U*u.Gyr + cosmo.lookback_time(rshift)),
                     zmax=1E6) for U in LBT]

    src_sfh = ColumnDataSource(data = {'LBT':LBT,'SFH':SFH, 'z':zs })
    
    sfh = figure(plot_width = 900, plot_height = 350, x_axis_label ='Lookback Time (Gyr)',
               y_axis_label = 'SFR (M/yr)')
    for i in range(90):
        sfh.line(SDB.LBT.values, SDB['{}_x_{}'.format(gid, i)].values, color = '#532436', alpha=.075)

    r1 = sfh.line(source = src_sfh, x = 'LBT', y='SFH', color = '#C1253C', line_width = 2)
    sfh.line(SDB.LBT,SDB['{}_16'.format(gid)], color ='black', line_width = 2)
    sfh.line(SDB.LBT,SDB['{}_84'.format(gid)], color ='black', line_width = 2)

    sfh.add_tools(HoverTool(tooltips = [('Lookback time', '@LBT'), 
                                        ('SFH', '@SFH'), ('z', '@z')], renderers = [r1]))
    sfh.xaxis.axis_label_text_font_size = "20pt"
    sfh.yaxis.axis_label_text_font_size = "20pt"
    sfh.xaxis.major_label_text_font_size = "15pt"
    sfh.yaxis.major_label_text_font_size = "15pt"
    
    return sfh
 
def Spec_plots(DB, BDB, RDB, PDB, MDB, gid):
    rshift = DB.query('id == {}'.format(gid)).zgrism.values[0]
    spec = MDB['{}_F'.format(gid)]
    wave = MDB.wave
    Pwv = PDB.wave
    Pflx = PDB['{}_F'.format(gid)]
    Perr = PDB['{}_E'.format(gid)]

    fmax = max(Pflx[Pflx**2>0]*1E18)
          
    try:
        Bfl = BDB['{}_F'.format(gid)]
        if max(Bfl[Bfl**2>0]*1E18) > fmax:
            fmax = max(Bfl[Bfl**2>0]*1E18)
    except:
        pass
    
    try:
        Rfl = RDB['{}_F'.format(gid)]
        if max(Rfl[Rfl**2>0]*1E18) > fmax:
            fmax = max(Rfl[Rfl**2>0]*1E18)
    except:
        pass
    
    S1 = figure(plot_width=900, plot_height=350, x_axis_label ='Wavelength (Å)',
        y_axis_label = 'F_λ  (10^-18)', tools = "tap,pan,wheel_zoom,box_zoom,reset",x_axis_type="log",
        x_range=(min(Pwv/(1+rshift))*0.95,max(Pwv/(1+rshift))*1.05), y_range=(0, fmax * 1.05))

    try:
        Bwv = BDB.wave
        Bfl = BDB['{}_F'.format(gid)]
        Ber = BDB['{}_E'.format(gid)]
        Bmfl = BDB['{}_M'.format(gid)]
        S1.circle(Bwv/(1+rshift), Bfl *1E18,color='#36787A',size = 3, alpha = 1, 
                  line_width = 1.5, line_color = 'black')
        S1.segment(Bwv/(1+rshift), (Bfl-Ber)*1E18, Bwv/(1+rshift),(Bfl+Ber)*1E18,color='#36787A')
        S1.line(Bwv[Bmfl**2>0]/(1+rshift), Bmfl[Bmfl**2>0]*1E18, color ='black', line_width = 2, alpha = 0.8)
    except:
        pass    
    try:
        Rwv = RDB.wave
        Rfl = RDB['{}_F'.format(gid)]
        Rer = RDB['{}_E'.format(gid)]
        Rmfl = RDB['{}_M'.format(gid)]
        S1.circle(Rwv/(1+rshift),Rfl*1E18,color='#EA2E3B',size = 3, alpha = 1, 
                  line_width = 1.5, line_color = 'black')
        S1.segment(Rwv/(1+rshift),(Rfl-Rer)*1E18,Rwv/(1+rshift),(Rfl+Rer)*1E18,color='#EA2E3B')
        S1.line(Rwv[Rmfl**2>0]/(1+rshift), Rmfl[Rmfl**2>0] *1E18, color ='black', line_width = 2, alpha = 0.8)
    except:
        pass    
    
    S1.circle(Pwv/(1+rshift),Pflx *1E18,color='#685877', size = 10, alpha = 1, 
              line_width = 1.5, line_color = 'black')
    S1.segment(Pwv/(1+rshift),(Pflx-Perr)*1E18,Pwv/(1+rshift),(Pflx+Perr)*1E18,color='#685877')
    S1.line(wave, spec*1E18, color ='black', alpha = 0.8)
    S1.xaxis.ticker = [2500,5000,10000,25000]
    S1.yaxis.axis_label_text_font_size=S1.xaxis.axis_label_text_font_size="20pt"
    S1.yaxis.major_label_text_font_size=S1.xaxis.major_label_text_font_size="15pt"
    
    return S1

def Table_plot(DB, gid):
    idx = DB.query('id == {}'.format(gid)).index.values[0]
    
    S1 = figure(plot_width=600, plot_height=225, x_range = (0,125)
                , y_range = (0,125), tools = '')

    box1 = BoxAnnotation(top=25, fill_alpha=0.1, fill_color='#EA2E3B')
    box2 = BoxAnnotation(bottom=25, top=50, fill_alpha=0.1, fill_color='#36787A')
    box3 = BoxAnnotation(bottom=50, top=75, fill_alpha=0.1, fill_color='#EA2E3B')
    box4 = BoxAnnotation(bottom=75, top=100, fill_alpha=0.1, fill_color='#36787A')
    box5 = BoxAnnotation(bottom=100, fill_alpha=0.1, fill_color='#EA2E3B')
    S1.add_layout(box1)
    S1.add_layout(box2)
    S1.add_layout(box3)
    S1.add_layout(box4)
    S1.add_layout(box5)

    cit1 = Label(x=2, y=102, text='{}-{}'.format(DB.field[idx], DB.id[idx]), 
                     render_mode='css', text_font_size = "22pt")
    cit2 = Label(x=2, y=77, text='z_grism = %1.3f' % (DB.zgrism[idx]), 
                     render_mode='css', text_font_size = "22pt")
    cit3 = Label(x=2, y=52, text='z_50 = %1.3f' % (DB.z_50[idx]), 
                     render_mode='css', text_font_size = "22pt")
    cit4 = Label(x=2, y=27, text='log(∑_1 (M_⊙ kpc^-2)) = %1.2f' % (np.log10(DB.Sigma1[idx])), 
                     render_mode='css', text_font_size = "22pt")
    cit5 = Label(x=2, y=2, text='log(M/M_⊙) = %1.2f' % (DB.lmass[idx]), 
                     render_mode='css', text_font_size = "22pt")
    S1.xgrid[0].grid_line_color=None
    S1.ygrid[0].grid_line_color=None

    S1.add_layout(cit1)
    S1.add_layout(cit2)
    S1.add_layout(cit3)
    S1.add_layout(cit4)
    S1.add_layout(cit5)

    S1.xaxis.major_label_text_color = S1.yaxis.major_label_text_color = None
    S1.yaxis.major_tick_line_color = S1.xaxis.major_tick_line_color =None 
    S1.yaxis.minor_tick_line_color = S1.xaxis.minor_tick_line_color =None  
    return S1

#for gid in [19850, 22358,23857,23758,23459,24177,27951,27185,27006,29879,37325,37210,40597,41147,42615,44133,47140]:
for idx in cats.DB.index:
    #idx = cats.DB.query('id == {}'.format(gid)).index.values[0]
    img = IMG_plot(cats.DB.id[idx])
    tbl = Table_plot(cats.DB, cats.DB.id[idx])
    sfh = SFH_plot(cats.DB, cats.SFH, cats.DB.id[idx])
    spec = Spec_plots(cats.DB, cats.g102, cats.g141, cats.phot, cats.model, cats.DB.id[idx])

    output_file('../../Vince-ec.github.io/appendix/sfhs/{}.html'.format(cats.DB.id[idx]),
                    title = '{}-{}'.format(cats.DB.field[idx], cats.DB.id[idx]))
    save(gridplot([[img, tbl], [sfh], [spec]]))