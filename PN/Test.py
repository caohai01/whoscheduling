import time
import pandas as pd
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from Utility import get_logger, get_real_data, read_validation_data, scale_data, samples2batch
import Config as cnf
from Nets import RecPointerNetwork
from Solution import BeamSearch, RunEpisode


def gr_inference(inst_data, start_time, dist_mat, args, run_episode):
    data_scaled = scale_data(inst_data, args)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():
        seq, _, _, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

    rewards = run_episode.mu.reward_fn(inst_data, seq, dist_mat)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route = [0] + [act.item() for act in seq] + [0]
    return route, score


def bs_inference(inst_data, start_time, dist_mat, args, run_episode):
    data_scaled = scale_data(inst_data, args)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    # beam_size = args.max_beam_size
    with torch.no_grad():
        seq, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')  # stochastic, greedy

    # 需要修改BeamSearch，处理seq去掉不必要的元素，否则计算长度会有问题
    seq_list = [seq[:, k] for k in range(seq.shape[1])]

    rewards = run_episode.mu.reward_fn(inst_data, seq_list, dist_mat)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route = [0] + [val.item() for val in seq[idx_max] if val.item() != 0] + [0]

    return route, score


def as_bs_inference(inp_data, args, run_episode, run_episode_bs):
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.learning_rate)

    inst_data, start_time, dist_mat = inp_data

    data_scaled = scale_data(inst_data, args)

    for _ in tqdm(range(args.train_epochs)):
        active_search_train_model(inst_data, data_scaled, start_time, dist_mat, run_episode, model_opt, args)

    # load previously training model
    run_episode_bs.neuralnet.load_state_dict(run_episode.neuralnet.state_dict())
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():
        seq, _ = run_episode_bs(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

    seq_list = [seq[:, k] for k in range(seq.shape[1])]
    rewards = run_episode.mu.reward_fn(inst_data, seq_list, dist_mat)

    max_reward, idx_max = torch.max(rewards, 0)

    score = max_reward.item()

    route = [0] + [val.item() for val in seq[idx_max] if val.item() != 0]
    route[-1] = 0

    return route, score


def run_single(inst_data, start_time, dist_mat, args, model, which_inf='bs'):
    tic = time.time()
    if which_inf == 'bs':
        run_episode_inf = BeamSearch(model, args).eval()
        route, score = bs_inference(inst_data, start_time, dist_mat, args, run_episode_inf)

    elif which_inf == 'gr':
        run_episode_inf = RunEpisode(model, args).eval()
        route, score = gr_inference(inst_data, start_time, dist_mat, args, run_episode_inf)

    elif which_inf == 'as_bs':

        run_episode_train = RunEpisode(model, args)

        run_episode_inf = BeamSearch(model, args).eval()
        inp_data = (inst_data, start_time, dist_mat)
        route, score = as_bs_inference(inp_data, args, run_episode_train, run_episode_inf)

    toc = time.time()

    output = dict([('score', score), ('route', route), ('time', toc - tic)])

    return output


def active_search_train_model(inst_data, data_scaled, inp_t_init_val, dist_mat, run_episode, model_opt, args):
    run_episode.train()
    binst_data, bdata_scaled = samples2batch(inst_data, data_scaled, args.batch_size)
    actions, log_prob, entropy, step_mask = run_episode(binst_data, bdata_scaled, inp_t_init_val, dist_mat,
                                                        'stochastic')
    rewards = run_episode.mu.reward_fn(inst_data, actions, dist_mat)
    av_rew = rewards.mean()
    advantage = (rewards - av_rew)
    res = advantage.unsqueeze(1) * log_prob + args.beta * entropy
    loss = -res[step_mask].sum() / args.batch_size
    model_opt.zero_grad()
    loss.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(run_episode.neuralnet.parameters(), args.max_grad_norm)
    model_opt.step()

if __name__ == "__main__":

    logger = get_logger(cnf.debug)

    # seed
    np.random.seed(cnf.seed)
    torch.manual_seed(cnf.seed)
    if str(cnf.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(cnf.seed)

    #  load dist data
    instance = f'{cnf.start_time}_{cnf.time_resolution}'
    cnf.instance = instance
    raw_data, raw_distm = get_real_data(cnf, instance)


    # load model
    saved_model_path = f'{cnf.save_w_dir}/model_{instance}_{cnf.saved_model_epoch}.pkl'
    logger.info(f'Loading model {saved_model_path} for instance {instance} ...')

    model = RecPointerNetwork(cnf.feature_count, cnf.dynamic_feature_count, cnf.rnn_hidden, cnf).to(cnf.device).eval()
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))

    # inference
    if not cnf.generated: # current data
        logger.info(f'Inferring route for current instance with {cnf.search_type}...')

        start_time = 0  # raw_data[0, cnf.OPENING_TIME_WINDOW_IDX]
        output = run_single(raw_data, start_time, raw_distm, cnf, model, which_inf=cnf.search_type)
        logger.info(10 * '-')
        logger.info(f'route: length: {len(output["route"])},  {output["route"]}')
        logger.info(f'total score: {output["score"]}')
        inference_time_ms = int(1000 * output['time'])
        logger.info(f'inference time: {inference_time_ms} ms')
        logger.info(10 * '-')
    else:
        inp_val = read_validation_data(cnf)
        logger.info(f'Inferring routes for {len(inp_val)} generated instances with {cnf.search_type}...')

        outputs = []
        for k, inst_data in enumerate(tqdm(inp_val)):
            start_time = 0  # inst_data[0: args.OPENING_TIME_WINDOW_IDX]
            output = run_single(inst_data, start_time, raw_distm, cnf, model, which_inf=cnf.search_type)
            outputs.append(output)

        df_out = pd.DataFrame(outputs)
        average_total_score = round(df_out.score.mean(), 2)
        average_time_ms = int(1000 * df_out.time.mean())
        logger.info(10 * '-')
        logger.info(f'search: {cnf.search_type} ')
        logger.info(f'average total score: {average_total_score}')
        logger.info(f'average inference time: {average_time_ms} ms')
        logger.info(10 * '-')
