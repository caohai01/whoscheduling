import os
from time import sleep

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
'''
input：center_ra，center_dec，radius，outfile，out_columns
'''
start_ra = 0
start_dec = -10
delta_ra = 4
delta_dec = 4
min_mag = 11.0
max_mag = 11.3
# radius = u.Quantity(1, u.deg)

out_columns = ['source_id ', 'ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']


Gaia.ROW_LIMIT = 5 # 返回无限数量的行集 Gaia.ROW_LIMIT为-1。
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
file_name = f"gaiadr3_{Gaia.ROW_LIMIT}_{delta_ra}x{delta_dec}_{min_mag}-{max_mag}.csv"

for dec in range(start_dec, 91, delta_dec):
    df_gs = pd.DataFrame(columns=out_columns)
    for ra in range(start_ra, 361, delta_ra):
        query = f"""SELECT TOP {Gaia.ROW_LIMIT} { ','.join(out_columns) } 
                    FROM gaiadr3.gaia_source gs
                    WHERE gs.ra >= {ra} AND gs.ra < {ra + delta_ra} 
                    AND gs.dec >= {dec} AND gs.dec < {dec + delta_dec} 
                    AND gs.phot_g_mean_mag <= {max_mag} AND gs.phot_g_mean_mag >= {min_mag}"""
        # coord = SkyCoord(ra=center_ra, dec=center_dec, unit=(u.degree, u.degree), frame='icrs')
        # j = Gaia.cone_search_async(coord, radius, columns=out_columns)
        try:
            r = Gaia.launch_job_async(query=query)
            table = r.get_results()
        except Exception as e:
            print(e)
            sleep(3)
            continue
        # r.pprint()
        df_gs = table.to_pandas()
        df_gs['snr'] = 100 # np.random.randint(1, 20, df_gs.shape[0]) * 5
        df_gs['filter_index'] = 5 # np.random.randint(0, 8, df_gs.shape[0])
        df_gs['visit_count'] = 3 # np.random.randint(1, 3, df_gs.shape[0])
        # print(len(table.items()))
        # df_gs.append(df, ignore_index=True)
        if not os.path.exists(file_name):
            df_gs.to_csv(file_name, header=True, index=False)
        else:
            df_gs.to_csv(file_name, mode='a', header=False, index=False)
