import glob
import os
import time

import numpy as np
import torch.cuda
from torch import optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
# from torch.nn import DataParallel
from torch.nn import utils
from tqdm import tqdm

import Config as cnf
from Nets import RecPointerNetwork
from Solution import RunEpisode
from Utility import *


logger = get_logger(True, __name__)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_step=5000):
    """Decay learning rate by a factor of 0.96 every lr_decay_epoch epochs.
       Lower_bounded at 0.00001"""
    lr = init_lr * (0.96 ** (epoch // lr_decay_step))
    if lr < 0.00001:
        lr = 0.00001

    if epoch % lr_decay_step == 0:
        logger.info('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train_model(raw_data, raw_dist_mat, episode, opt, config):
    bnew_list = []
    bnew_scaled_list = []
    for _ in range(config.batch_size):
        new_data = sample_random_data(raw_data, config)
        new_data_scaled = scale_data(new_data, config)
        bnew_list.append(new_data)
        bnew_scaled_list.append(new_data_scaled)

    bnew_data = torch.stack(bnew_list, dim=0)
    bnew_data_scaled = torch.stack(bnew_scaled_list, dim=0)
    # new_data = sample_random_data(raw_data, config)
    # new_data_scaled = scale_data(new_data, config)
    # bnew_data, bnew_data_scaled = samples2batch(new_data, new_data_scaled, config.batch_size)

    episode.train()
    opt.zero_grad()
    dist_mat = raw_dist_mat
    start_time = 0
    actions, log_prob, entropy, step_mask = episode(bnew_data, bnew_data_scaled, start_time, dist_mat, 'stochastic')

    if len(actions) < 2:
        return 0, 0, 0, 0

    rewards = episode.mu.reward_fn(new_data, actions, dist_mat)

    # loss = 0
    mean_reward = rewards.mean()
    # median_reward = rewards.median()
    min_reward = rewards.min()
    max_reward = rewards.max()

    base_line = mean_reward # * 0.8 + max_reward * 0.2

    advantage = (rewards - base_line)  #advantage

    res = advantage.unsqueeze(1) * log_prob + config.beta * entropy

    loss = -res[step_mask].sum() / config.batch_size  # step_masked res == 0, remove them
    loss.backward(retain_graph=False)
    utils.clip_grad_norm_(episode.neuralnet.parameters(), config.max_grad_norm)

    opt.step()

    return mean_reward.item(), min_reward.item(), max_reward.item(), loss.item()


def test_model(data, start_time, dist_mat, episode, config):
    with torch.no_grad():
        data_scaled = scale_data(data, config)
        bdata, bdata_scaled = data.unsqueeze(0), data_scaled.unsqueeze(0)
        actions, log_prob, entropy, step_mask = episode(bdata, bdata_scaled, start_time, dist_mat, 'greedy')

        reward = episode.mu.reward_fn(data, actions, dist_mat)

        return reward.item()


def validation(inp_val, dist_mat, episode, config):
    reward_val = torch.tensor(0.0).to(config.device)
    rew_dict = {}
    for k, data in enumerate(inp_val):
        inst_data, start_time = data, 0
        rew = test_model(inst_data, start_time, dist_mat, episode, config)
        reward_val += rew
        key_str = 'a_' + str(k)
        rew_dict[key_str] = rew

    return rew_dict, reward_val.item() / len(inp_val)


def train_loop(val_data_array, raw_data, raw_dist_mat, episode, config):
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=config.learning_rate)
    step_dict = {}
    # torch.autograd.set_detect_anomaly(True)  # for debugging, winn slow down the training
    start_epoch = 1
    if cnf.continue_train:
        saved_models = glob.glob(f'{config.save_w_dir}/model_{instance}_*.pkl')
        if len(saved_models) > 0:
            # retrieve the wilde character part of the file name
            saved_epochs = [int(model_path.split('_')[-1].split('.')[0]) for model_path in saved_models]
            saved_epochs.sort()
            start_epoch = saved_epochs[-1] + 1
            model_path = f'{config.save_w_dir}/model_{instance}_{start_epoch-1}.pkl'
            logger.info(f'loading model from: {model_path}')
            run_episode.neuralnet.load_state_dict(torch.load(model_path, map_location=cnf.map_location, weights_only=False))

    for epoch in tqdm(range(start_epoch, cnf.train_epochs + 1)):

        avg_reward, min_reward, max_reward, loss = train_model(raw_data, raw_dist_mat, episode, model_opt, config)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        exp_lr_scheduler(model_opt, epoch, init_lr=config.learning_rate)

        if epoch == 0 or epoch % config.print_epochs == 0:

            logger.info(f'Epoch {epoch} completed')
            logger.info(f'validating current instance data ...')
            rew_dict, avg_reward_inst = validation([raw_data], raw_dist_mat, run_episode, config)
            #logger.info(f'validating test data ...')
            #avg_reward_valid = -1
            logger.info(f'validating valid data ...')
            _, avg_reward_valid = validation(val_data_array, raw_dist_mat, run_episode, config)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_inst, avg_reward_valid, loss_total])

            else:
                avg_loss = loss_total / config.print_epochs
                avg_reward_total = reward_total / config.print_epochs
                avg_min_reward_total = min_reward_total / config.print_epochs
                avg_max_reward_total = max_reward_total / config.print_epochs

                history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_inst, avg_reward_valid, avg_loss])

            logger.info(f'Average total loss: {avg_loss: 2.6f}')
            logger.info(f'Average train mean reward: {avg_reward_total: 2.6f}')
            logger.info(f'Average train max reward: {avg_max_reward_total: 2.6f}')
            logger.info(f'Average train min reward: {avg_min_reward_total: 2.6f}')
            logger.info(f'Validation average reward on valid: {avg_reward_valid: 2.6f}')
            logger.info(f'Validation reward on instance: {avg_reward_inst: 2.6f}')

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % cnf.save_epochs == 0 and not cnf.debug:
            if cnf.parallel_train and torch.distributed.get_rank() != 0: # save only in master gpu
                continue
            model_path = f'{config.save_w_dir}/model_{instance}_{epoch}.pkl'
            logger.info(f'saving model to: {model_path}')
            # logger.handlers[0].flush()
            torch.save(run_episode.neuralnet.state_dict(), model_path)

    return history


def save_training_history(training_history, file_path):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
                          'min_reward_train', 'max_reward_train', 'avg_reward_val',
                          'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns=TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


if __name__ == '__main__':

    using_gpu = 'cuda' in str(cnf.device)
    gpu_count = torch.cuda.device_count()
    logger.info(f'found {gpu_count} GPUs, train on {cnf.cuda_id} with device: {cnf.device}')

    # 随机种子，为了重现结果
    np.random.seed(cnf.seed)
    torch.manual_seed(cnf.seed)
    if using_gpu:
        torch.cuda.manual_seed(cnf.seed)

    # get val and real instance scores
    instance = f'{cnf.start_time}_{cnf.time_resolution}'
    logger.info(f'instance : {instance}, continue train: {cnf.continue_train}')
    cnf.instance = instance
    inp_real, inp_dist = get_real_data(cnf, instance)
    # 生成验证数据并保存
    generate_validation_data(inp_real, cnf)
    # 读取保存的验证数据
    val_array = read_validation_data(cnf)

    # save config to file
    save_config(cnf)
    # invalid,Expected parameter probs (Tensor of shape (32, 62)) of distribution Categorical(probs: torch.Size([32, 62])) to satisfy the constraint Simplex(), but found invalid values:
    # Distribution.set_default_validate_args(False)

    model = RecPointerNetwork(cnf.feature_count, cnf.dynamic_feature_count, cnf.rnn_hidden, cnf).to(cnf.device)
    if using_gpu and gpu_count > 1 and cnf.parallel_train:
        dist.init_process_group(backend='nccl')
        logger.info(f'Using DistributedDataParallel mode')
        model = DistributedDataParallel(model)

    run_episode = RunEpisode(model, cnf)

    training_history = train_loop(val_array, inp_real, inp_dist, run_episode, cnf)

    if using_gpu and gpu_count > 1 and cnf.parallel_train:
        # 清理
        dist.destroy_process_group()

    # save history
    if not cnf.debug:
        file_path = f'{cnf.save_h_dir}/train_his_{instance}_{cnf.train_epochs}.csv'
        save_training_history(training_history, file_path)
