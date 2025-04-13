import torch
import os

data_path = "input"  # 数据存放路径
# data_{instance}.csv为基本数据
# dist_{instance}.csv为距离数据
X_COORDINATE_IDX = 0
Y_COORDINATE_IDX = 1
FILTER_IDX = 2
VIS_DURATION_TIME_IDX = 2
SCORE_2_IDX = 3
SCORE_1_IDX = 4
SCORE_0_IDX = 5
COEF_TIMES = 1e5  # 为了在输入文件中保持精度，计算时除以该值
# REWARD_K_IDX = 3
# K_TIMES = 1e5  # 为了在文件中保持精度
# REWARD_B_IDX = 4
# B_TIMES = 1e2 # 为了在文件中保持精度
RISE_TIME_WINDOW_IDX = 6
SET_TIME_WINDOW_IDX = 7
ARRIVAL_TIME_IDX = 8


output_directory = "output"
val_set_pt_file = "val_set.pt"
save_w_dir = "output/models"
save_h_dir = "output/models"
sample_count = 64

cuda_id = 0
device = torch.device('cpu')
map_location = {'cpu': 'cpu'}
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(cuda_id)
    map_location = {'cpu': f'cuda'}
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 为了避免cuda内存碎片化，内存不足时开启

feature_count = 9 # 特征数量
dynamic_feature_count = 9 # 动态特征数量
rnn_hidden = 128 # 隐藏曾
n_layers = 2
n_heads = 8  # 多头注意力
ff_dim = 256  # feed forward
use_checkpoint = False
parallel_train = False  # 是否使用多个GPU训练
continue_train = False  # 是否继续训练

batch_size = 32
learning_rate = 1e-4
beta = 1e-2
max_grad_norm = 1.0

train_epochs = 1000
save_epochs = 100
print_epochs = 100

debug = False
seed = 1


start_time = "0125140000"
time_resolution = "5s"
# instance = {start_time}_{time_resolution}

# test config
generated = False
search_type = 'gr'  # bs, gr, as_bs
max_beam_width = 9
saved_model_epoch = 10000

# comment
comment = 'completion, score difference at time'
