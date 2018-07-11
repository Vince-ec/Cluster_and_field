import pandas as pd
from spec_exam import Gen_DB_and_beams

s_cand = pd.read_pickle('../dataframes/galaxy_frames/s_candidates.pkl')
n_cand = pd.read_pickle('../dataframes/galaxy_frames/n_candidates.pkl')

#for i in s_cand.index:
#    Gen_DB_and_beams(s_cand.gids[i], 'south', s_cand.ra[i], s_cand.dec[i])
    
for i in n_cand.index:
    Gen_DB_and_beams(n_cand.gids[i], 'north', n_cand.ra[i], n_cand.dec[i])