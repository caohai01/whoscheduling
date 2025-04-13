# 转换为小光电支持的格式jhj
import json
from datetime import datetime
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
import numpy as np
import glob
import astroschedule.Config as cnf
import os.path as path
result_types = ["Unknown", "Random", "Greedy", "MAX", "VNS", "RL"]
itype = 2
sched_name = result_types[itype]
target_file = path.basename(cnf.target_file)

file_pattern = (f"../plan_{sched_name}_{cnf.start_time.strftime('%Y%m%d%H%M')}_{cnf.time_duration.to(u.hour).value:.1f}"
                f"_*_{target_file}")
file_list = glob.glob(file_pattern)
source = file_list[0]
jhj_lst = []
from_time = None
trans_start_time = None
with open(source) as f:
    while True:
        line = f.readline()
        if not line or len(line) < 2:
            break
        line = line[:-2]  # 去掉逗号和换行
        # 解析成 dict对象
        item = json.loads(line)
        target = item['target']
        if target == 'transition':  # 转换动作，仅记录开始时间
            trans_start_time = datetime.strptime(item['start_time'], '%Y-%m-%d %H:%M:%S')
            continue
        ra = item['RA']
        dec = item['Dec'] + 4.8 # +4.8 均指向下半部分，对应2和3，-4.8指向上半部分，对应1和4
        start_time = datetime.strptime(item['start_time'], '%Y-%m-%d %H:%M:%S')
        if trans_start_time is not None:
            if trans_start_time < start_time:
                start_time = trans_start_time
            trans_start_time = None
        if from_time is None:  # 文件名用
            from_time = start_time
        end_time = datetime.strptime(item['end_time'], '%Y-%m-%d %H:%M:%S')
        exp_time = item['exposure_duration']
        filter_position = item['filter_index'][0]
        jhd = f'{sched_name}_{target}.JHD'
        if len(jhd) > 30:
            jhd = jhd[:30]
        jhj_lst.append([str(itype) + target[-5:], start_time.strftime('%Y%m%d %H%M%S'),
                        end_time.strftime('%Y%m%d %H%M%S'),
                        ra, ra,
                        dec, dec,
                        exp_time,
                        jhd,
                        int(round(exp_time * 1000)),
                        1,
                        filter_position,
                        2])

jhj_formats = {'region': '%6d', 'start_time': '%15s', 'end_time': '%15s', 'start_hour1': '%6.1f',
               'end_hour1': '%6.1f', 'start_hour2': '%5.1f', 'end_hour2': '%5.1f',
               'magitude': '%4.1f', 'file': '%30s', 'exposure_duration': '%05d', 'mode': '%03d', 'filter': '%1d',
               'type': '%1d'}
jhj_types = ('i8', 'S', 'S', 'f8', 'f8', 'f8', 'f8', 'f8', 'S', 'i4', 'i4', 'i4', 'i4')

jhj_data = np.array(jhj_lst)  # jhj_list为观测数据值List
tb_jhj = Table(jhj_data, names=jhj_formats.keys(), dtype=jhj_types)
file_name = f"xgd_{sched_name}_{from_time.strftime('%Y%m%d%H%M')}_{jhj_data.shape[0]}.JHJ"
ascii.write(tb_jhj, file_name, overwrite=True, format='fixed_width_no_header', delimiter=None, formats=jhj_formats)
print(file_name)