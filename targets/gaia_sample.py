import pandas as pd
import numpy as np

# Load your data into a DataFrame (example: df)
df = pd.read_csv('gaia_5_4x4_10-13.csv')

# Randomly select N stars
# SOURCE_ID,ra,dec,phot_g_mean_mag,snr,filter_index,visit_count
N = 300
rnd_seed = 1
np.random.seed(rnd_seed)
df = df[df['phot_g_mean_mag'] < 13]
df = df[df['phot_g_mean_mag'] >= 10]
random_stars = df.sample(n=N)

# target_id,RA,Dec,mag1,mag2,author,snr
random_stars.columns = ['target_id', 'RA', 'Dec', 'mag1', 'snr', 'filter_index', 'visit_count']
random_stars.drop(['filter_index', 'visit_count'], axis=1, inplace=True)
random_stars['mag2'] = random_stars['mag1']
random_stars['author'] = 'ch'
random_stars['snr'] = 100
random_stars['count'] = 3

random_stars.to_csv(f'gaia_{N}.csv', header=True, index=False)
print(f'gaia_{N}.csv')