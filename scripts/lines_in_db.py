#!/home/vestrada78840/miniconda3/envs/astroconda/bin/python
from spec_id import *
from spec_id_2d import *
import numpy as np
import fsps

full_db = pd.read_pickle(data_path + 'all_galaxies_1d.pkl')

full_db['Hd'] = np.repeat(-99, len(full_db))
full_db['Hg'] = np.repeat(-99, len(full_db))
full_db['Hb'] = np.repeat(-99, len(full_db))
full_db['Ha'] = np.repeat(-99, len(full_db))
full_db['OII'] = np.repeat(-99, len(full_db))
full_db['OIII'] = np.repeat(-99, len(full_db))
full_db['SII'] = np.repeat(-99, len(full_db))

full_db['Hd_sig'] = np.repeat(-99, len(full_db))
full_db['Hg_sig'] = np.repeat(-99, len(full_db))
full_db['Hb_sig'] = np.repeat(-99, len(full_db))
full_db['Ha_sig'] = np.repeat(-99, len(full_db))
full_db['OII_sig'] = np.repeat(-99, len(full_db))
full_db['OIII_sig'] = np.repeat(-99, len(full_db))
full_db['SII_sig'] = np.repeat(-99, len(full_db))

for i in full_db.index:
    try:
        ldb = np.load(pos_path + '{}_{}_line_fits.npy'.format(field, galaxy), allow_pickle = True).item()
        full_db['Hd'][i] = ldb['Hd']
        full_db['Hg'][i] = ldb['Hg']
        full_db['Hb'][i] = ldb['Hb']
        full_db['Ha'][i] = ldb['Ha']
        full_db['OII'][i] = ldb['OII']
        full_db['OIII'][i] = ldb['OIII']
        full_db['SII'][i] = ldb['SII']

        full_db['Hd_sig'][i] = ldb['Hd_sig']
        full_db['Hg_sig'][i] = ldb['Hg_sig']
        full_db['Hb_sig'][i] = ldb['Hb_sig']
        full_db['Ha_sig'][i] = ldb['Ha_sig']
        full_db['OII_sig'][i] = ldb['OII_sig']
        full_db['OIII_sig'][i] = ldb['OIII_sig']
        full_db['SII_sig'][i] = ldb['SII_sig']
    except:
        pass
full_db.to_pickle(data_path + 'all_galaxies_wlines_1d.pkl')