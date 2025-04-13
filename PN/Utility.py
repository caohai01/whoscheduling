import json
import logging

import pandas as pd
import torch


def get_logger(debug=False, name=None):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    # logging.basicConfig(format='%(message)s', level=logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def save_config(config):
    keys_list = ['instance', 'n_layers', 'n_heads', 'ff_dim', 'feature_count',
                 'dynamic_feature_count', 'rnn_hidden', 'learning_rate', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'save_epochs', 'train_epochs', 'comment']

    dict_to_save = {key: config.__dict__[key] for key in keys_list}

    with open(config.save_w_dir + '/model_training_args.txt', 'w') as f:
        json.dump(dict_to_save, f)

    return dict_to_save


def load_saved_config(config):
    with open(config.load_w_dir + '/model_training_args.txt') as json_file:
        data = json.load(json_file)
        config.n_layers = data['n_layers']
        config.n_heads = data['n_heads']
        config.ff_dim = data['ff_dim']
        config.feature_count = data['feature_count']
        config.dynamic_feature_count = data['dynamic_feature_count']
        config.rnn_hidden = data['rnn_hidden']

    return config


def get_real_data(config, instance):
    data_path = f"{config.data_path}/data_{instance}.csv"
    dataf = pd.read_csv(data_path,
                        usecols=['RA(deg)', 'Dec(deg)', 'duration', 'score_a2(1e5)', 'score_a1(1e5)', 'score_a0(1e5)',
                                 'rise_time', 'set_time'])
    dist_path = f"{config.data_path}/dist_{instance}.csv"
    distf = pd.read_csv(dist_path, header=None)
    # 增加一列totalTime，让它等于第一行的总时间
    TOTAL_TIME_KEY = 'Total Time'
    dataf[TOTAL_TIME_KEY] = 0
    dataf[TOTAL_TIME_KEY] = dataf.loc[0]['set_time']

    data = dataf.values
    dist = distf.values
    return torch.FloatTensor(data).to(config.device), torch.FloatTensor(dist).to(config.device)


def sample_random_data(data, config):
    device = config.device
    zero_scalar = torch.zeros(1, dtype=torch.float, device=device)
    new_data = data.clone()

    COL_X = config.X_COORDINATE_IDX
    COL_Y = config.Y_COORDINATE_IDX
    # COL_K = config.REWARD_K_IDX
    # COL_COEF2 = config.SCORE_2_IDX
    # COL_COEF1 = config.SCORE_1_IDX
    # COL_COEF0 = config.SCORE_0_IDX
    COL_COEFS = [config.SCORE_2_IDX, config.SCORE_1_IDX, config.SCORE_0_IDX]
    COEF_TIMES = torch.tensor(config.COEF_TIMES).to(device)

    # K_TIMES = torch.tensor(config.K_TIMES).to(device)
    # COL_B = config.REWARD_B_IDX
    # B_TIMES = torch.tensor(config.B_TIMES).to(device)

    margin = torch.tensor(1.0).to(device)

    COL_OPENING = config.RISE_TIME_WINDOW_IDX
    COL_CLOSING = config.SET_TIME_WINDOW_IDX
    COL_DURATION = config.VIS_DURATION_TIME_IDX

    start_time = data[0, COL_OPENING]
    end_time = data[0, COL_CLOSING]
    n_count = data.shape[0]

    coefs = data[1:, COL_COEFS] # the first node is the park node

    max_c = coefs.max(dim=0)[0]
    min_c = coefs.min(dim=0)[0]
    range_c = max_c - min_c
    max_c += range_c * margin
    max_c[0] = torch.min(max_c[0], zero_scalar) # 处理二次系数大于0的情况，保证a < 0
    min_c -= range_c * margin
    range_c = max_c - min_c
    rnd_coefs = torch.rand(n_count, range_c.shape[0], device=device) * range_c + min_c

    a = rnd_coefs[:, 0] / COEF_TIMES
    b = rnd_coefs[:, 1] / COEF_TIMES
    c = rnd_coefs[:, 2] / COEF_TIMES
    da = 2.0 * a
    xp = - b / da
    t = 10
    while t > 0:
        t = t - 1
        delta = (b ** 2 - 4.0 * a * c)
        if (delta < 0).any():
            # 如果c < 0, 则增加常数项的值
            # 处理二次函数顶点在0以下的情况，增加常数项的值
            rnd_coefs[delta < 0, 2] += 0.1 * COEF_TIMES
            c = rnd_coefs[:, 2] / COEF_TIMES
        else:
            break

    new_data[:, COL_COEFS] = rnd_coefs

    # 根据二次函数系数确定rise(open)和set(close)
    delta_root = torch.sqrt(delta)
    x1 = (xp + delta_root / da) # 左侧
    x1[x1 < start_time] = start_time
    x1[x1 > end_time] = end_time

    x2 = (xp - delta_root / da) # 右侧
    x2[x2 > end_time] = end_time
    x2[x2 < start_time] = start_time

    new_data[:, COL_OPENING] = x1
    new_data[:, COL_CLOSING] = x2

    # low_b = min_b - (max_b - min_b) * margin
    # high_b = max_b + ((max_b - min_b) * margin)
    #
    # max_k = k.max()
    # min_k = k.min()
    #
    # low_k = min_k - (max_k - min_k) * margin
    # high_k = max_k + ((max_k - min_k) * margin)
    #
    # # random scores
    # rnd_bs = torch.rand(n, device=device) * (high_b - low_b) + low_b
    # rnd_ks = torch.rand(n, device=device, ) * (high_k - low_k) + low_k

    # clip by 0 = b + kx, x = -b / k
    # clip by 100 = b + kx, x = 1.0 - b / k
    # score_max = torch.tensor(1.0, device=device)
    # new_data[:, COL_OPENING] = -rnd_bs / rnd_ks
    # new_data[:, COL_CLOSING] = (score_max - rnd_bs) / rnd_ks
    #
    # new_data[:, COL_B] = rnd_bs * B_TIMES
    # new_data[:, COL_K] = rnd_ks * K_TIMES
    # 处理随机分数系数中最大值小于0的情况
    # new_data[:, COL_COEFS] = rnd_coefs

    # # 处理那些不合理的 起始结束时间
    # reversed_condition = new_data[:, COL_OPENING] > new_data[:, COL_CLOSING]
    # reversed_indices = torch.nonzero(reversed_condition).squeeze()
    # org_opening = new_data[reversed_indices, COL_OPENING].clone()
    # new_data[reversed_indices, COL_OPENING] = new_data[reversed_indices, COL_CLOSING]
    # new_data[reversed_indices, COL_CLOSING] = org_opening
    #
    # new_data[:, COL_OPENING] = torch.ceil(torch.clamp(new_data[:, COL_OPENING], start_time, end_time))
    # new_data[:, COL_CLOSING] = torch.ceil(torch.clamp(new_data[:, COL_CLOSING], start_time, end_time))

    # start_time = new_data[:, COL_OPENING].min()
    # end_time = new_data[:, COL_CLOSING].max()

    new_data[0, COL_OPENING] = new_data[:, COL_OPENING].min()
    new_data[0, COL_CLOSING] = new_data[:, COL_CLOSING].max()

    # # 随机移动park位置，赤经[0, 360]，赤纬[-30, 90]
    # park_ra, park_dec = torch.rand(2, device=device)
    # new_data[0, COL_X] = park_ra * 360.0
    # new_data[0, COL_Y] = park_ra * 120.0 - 30.0
    # xy_range = torch.tensor([360.0, 120.0], device=device)
    # xy_offset = torch.tensor([0.0, 30.0], device=device)
    # new_data[0, COL_X], new_data[0, COL_Y] = torch.rand(2, device=device) * xy_range - xy_offset
    # 不再以最后一行作为结束标志
    # new_data[-1] = new_data[0].clone()

    # # 重新计算距离矩阵，由于矩阵非常复杂，暂时不好计算，使用缺省值
    # # 能不能使用欧式距离估计值？
    # dist = torch.zeros((1, n), device=device)
    # for i in range(1, n-1):
    #     dist[0, i] = get_dist(data[0] - data[i])
    # dist_mat[0] = dist
    # dist_mat[-1] = dist.clone()

    # 经历时间在原有时间基础上随机增加或减少10%
    new_data[:, COL_DURATION] = data[:, COL_DURATION] * (1.0 + 0.2 * (torch.rand(n_count, device=device) - 0.5))
    return new_data


def read_validation_data(config):
    inp_val_path = f'{config.output_directory}/{config.val_set_pt_file}'
    val_data = torch.load(inp_val_path, map_location=config.map_location, weights_only=True)
    return val_data


def generate_validation_data(raw_data, config):
    inp_val = [sample_random_data(raw_data, config) for _ in range(config.sample_count)]
    output_path = f'{config.output_directory}/{config.val_set_pt_file}'
    torch.save(inp_val, open(output_path, 'wb'))


def scale_data(data, config):
    min_vals = data.min(dim=0, keepdim=True)[0]
    max_vals = data.max(dim=0, keepdim=True)[0]
    # 按列归一化到 [0, 1]
    normalized_data = (data.clone() - min_vals) / (max_vals - min_vals)
    # 处理最大值与最小值相等的情况
    # 如果 max_vals 等于 min_vals，则将该列的值设为 1.0
    normalized_data[:, (max_vals == min_vals).squeeze()] = 1.0

    # tmax = data[0, config.CLOSING_TIME_WINDOW_IDX]
    # # min_vals, _ = torch.min(data, dim=0, keepdim=True)
    # max_vals, _ = torch.max(data, dim=0, keepdim=True)
    # max_vals[:, config.OPENING_TIME_WINDOW_IDX] = tmax
    # max_vals[:, config.CLOSING_TIME_WINDOW_IDX] = tmax
    # max_vals[:, config.ARRIVAL_TIME_IDX] = tmax
    #normalized_data = data.clone() / max_vals

    return normalized_data


def samples2batch(new_data, new_data_scaled, batch_size):
    bnew_data = new_data.expand(batch_size, -1, -1)
    bnew_data_scaled = new_data_scaled.expand(batch_size, -1, -1)
    return bnew_data, bnew_data_scaled
