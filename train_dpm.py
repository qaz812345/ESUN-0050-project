
import os
import time
import math
import json
import pickle
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

import torch
import torch.nn as nn

from env import *
from models import Policy

import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='PyTorch DPM Args')
# training options
parser.add_argument('--model', type=str, default='dpm',
                    help='model: dpm (default: dpm)')
parser.add_argument('--w_init', type=str, default='default',
                    help='weight initial method = [default, equal, random]')
parser.add_argument('--input_shift', action='store_true', default=False,
                    help='model input minus one (default: False)')
parser.add_argument('--rm_smooth_cost', action='store_true', default=False,
                    help='use smooth cost (default: False)')
parser.add_argument('--rm_money_reg', action='store_true', default=False,
                    help='remove regularization panelty on money (default: False)')
parser.add_argument('--rolling', action='store_true', default=False,
                    help='use rolling in validation and test (default: False)')
parser.add_argument('--num_rolling_steps', type=int, default=10,
                    help='maximum number of rolling steps (default: 10)')
parser.add_argument('--geo_prob', type=float, default=5e-3,
                    help='probability of geometric distribution (default: 5e-3)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
# hyperparameters
parser.add_argument('--lr', type=float, default=3e-6,
                    help='learning rate (default: 3e-6)')
parser.add_argument('--num_steps', type=int, default=500,
                    help='number of epochs (default: 500)')
parser.add_argument('--n_episode', type=int, default=128,
                    help='number of episode (default: 128)')
parser.add_argument('--episode_step', type=int, default=50,
                    help='number of steps (default: 50)')
parser.add_argument('--win_size', type=int, default=31,
                    help='sliding window size (default: 31)')
parser.add_argument('--cash_reward', type=str, default='default',
                    help='cash reward mode = [default, custom]')
parser.add_argument('--cr_scale', type=int, default=5,
                    help='cash reward scale (default: 5)')
parser.add_argument('--cr_shift', type=float, default=1.004,
                    help='cash reward shift (default: 1.004)')
parser.add_argument('--cnn_d', type=int, default=10,
                    help='cnn dimension (default: 10)')
parser.add_argument('--reg_w', type=float, default=1e-4,
                    help='max weight regularization scale (default: 1e-4)')

# GPU option
parser.add_argument('--no_cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
parser.add_argument('--cuda_no', type=int, default=0,
                    help='cuda no (default: 0)')
# batch of experiment name
parser.add_argument('--file_name', type=str, default='results_0725',
                    help='file name (default: results)')
args = parser.parse_args()


action_n = 11
window_size = args.win_size
episode_step = args.episode_step
num_steps = args.num_steps
rolling = args.rolling
num_rolling_steps = args.num_rolling_steps

if args.w_init == 'default':
    init_w = np.array(np.array([1]+[0]*(action_n -1)))
elif args.w_init == 'equal':
    init_w = np.array(np.array([1/(action_n)] * (action_n)))
elif args.w_init == 'random':
    temp = torch.randint(1, 100, (11,))
    init_w = temp / temp.sum()

if args.seed != -1:
    torch.manual_seed(args.seed)

if args.cash_reward == 'default':
    cr_mode = 'default'
else:
    cr_mode = '{}x{}'.format(args.cr_shift, args.cr_scale)

if args.rm_money_reg:
    expReg_mode = '_woM'
else:
    expReg_mode = ''

if args.rm_smooth_cost:
    nc_mode = '-woRs'
else:
    nc_mode = ''

# set device
useCuda = not args.no_cuda
device = torch.device("cuda:{}".format(args.cuda_no) if (torch.cuda.is_available() and useCuda) else "cpu")

def main():
    # step 1. Set saving directory
    results_dict = {'l1_losses': [],
                    'train_smooth_rewards': [],
                    'train_rewards': [],
                    'train_smooth_rewards_eq': [],
                    'train_rewards_eq': [],
                    'val_rolling_action': [],
                    'val_rolling_pv': [],
                    'val_rolling_pv_eq': [],
                    'val_rolling_rewards': [],
                    'val_rolling_rewards_eq': [],
                    'test_rolling_pv': [],
                    'test_rolling_pv_eq': [],
                    'test_rolling_action': [],
                    'test_rolling_rewards': [],
                    'test_rolling_rewards_eq': [],
                    'val_scores':[],
                    'val_action':[],
                    'val_pv': [],
                    'val_pv_eq': [],
                    'test_action':[],
                    'test_pv': [],
                    'test_pv_eq': []
                    }

    base_dir = (os.getcwd() + '/models/' + args.file_name + '-seed_' + str(args.seed) +
                                                            '-base-smooth_1e-3-reg-oldupdate/' + 
                                                            'e_' + str(args.num_steps) +
                                                            '-b_' + str(args.n_episode) +
                                                            'x' + str(args.episode_step) +
                                                            '-w_' + str(args.win_size) +
                                                            '-lr_' + str(args.lr) + 
                                                            '-rollep_' + str(args.num_rolling_steps) +
                                                            '-reg_w_' + str(args.reg_w) +
                                                            '-cr_' + cr_mode + 
                                                            '-gp_' + str(args.geo_prob) + 
                                                            '/pid_' + str(os.getpid()) + '/')
    exp_name = base_dir.split('/')[-3]

    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(base_dir + '/ckpt'):
        os.mkdir(base_dir + '/ckpt')

    with open(base_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f)


    # step 2. Read data
    path_train_data = './np_data/s10_f5_train_9501_0412.npy'
    path_test_data = './np_data/s10_f5_test_0501_0812.npy'
    train_data = np.load(path_train_data) #[F, S, T]
    test_data = np.load(path_test_data)
    data = np.concatenate((train_data, test_data), axis=2)
    val_steps = 300
    train_steps = train_data.shape[2] - val_steps
    test_steps = test_data.shape[2]

    env = TradeEnv(data=data, window_length=window_size, n_batch=args.n_episode,
                portfolio_value=10000, trading_cost=[0.1425/100, 0.4425/100],
                interest_rate=0)
    env_test = TradeEnv(data=data, window_length=window_size,
                portfolio_value=10000, trading_cost=[0.1425/100, 0.4425/100],
                interest_rate=0)
    env_eq = TradeEnv(data=data, window_length=window_size,
                portfolio_value=10000, trading_cost=[0.1425/100, 0.4425/100],
                interest_rate=0)

    # step 3. define model
    pi = Policy(train_data.shape[0], action_n, args.lr, args.cnn_d, args.input_shift, window_size).to(device)


    def draw(total_steps, size, window_length=window_size, batch_size=episode_step, beta=args.geo_prob):
        ''' Sample valid start index from geometric distribution with probability beta'''
        print('range:', total_steps)
        i_start_list = list()
        for _ in range(size):
            while 1:
                z = np.random.geometric(p=beta)
                i = total_steps - batch_size - 1 - z
                if (i > window_length) and i < (total_steps - batch_size - 1) and (i not in i_start_list):
                    i_start_list.append(i)
                    break
        return np.array(i_start_list)

    def eval(mode, e=None, rolling=False, save_value=False):
        ''' Portfolio value evaluation '''
        max_pv = 10000
        pv_ratio = []
        if mode =='test':
            steps = test_steps
            start_index = train_steps + val_steps
        if mode == 'val':
            steps = val_steps
            start_index = train_steps

        w_eq = np.array(np.array([1/(action_n)] * (action_n)))
        state_test, done_test = env_test.reset(init_w, 10000, t=start_index)
        state_eq, done_eq = env_eq.reset(w_eq, 10000, t=start_index)
        pv_eq = state_eq[2]
        
        # init values
        if save_value:
            if mode == 'test'and not rolling:
                results_dict['test_pv'].append(state_test[2])
                results_dict['test_pv_eq'].append(state_eq[2])
                results_dict['test_action'].append(state_test[1])
            elif mode == 'val' and not rolling:
                results_dict['val_pv'].append(state_test[2])
                results_dict['val_pv_eq'].append(state_eq[2])
                results_dict['val_action'].append(state_test[1])
            elif mode == 'test' and rolling:
                results_dict['test_rolling_pv'].append(state_test[2])
                results_dict['test_rolling_pv_eq'].append(state_eq[2])
                results_dict['test_rolling_action'].append(state_test[1])
            elif mode == 'val' and rolling:
                results_dict['val_rolling_pv'].append(state_test[2])
                results_dict['val_rolling_pv_eq'].append(state_eq[2])
                results_dict['val_rolling_action'].append(state_test[1])
            
        for i in range(steps):
            pv_pre = state_test[2]
            x = torch.from_numpy(state_test[0]).to(device).float().unsqueeze(0)
            last_w = torch.from_numpy(state_test[1][1:]).to(device).float().unsqueeze(0)
            with torch.no_grad():
                action = pi(x, last_w)
            state_test, reward_test, done_test = env_test.step(action.squeeze(0).detach().cpu().numpy())
            state_eq, reward_eq, done_eq = env_eq.step(w_eq)
            last_w = state_test[1]
            pv_cur = state_test[2]
            pv_ratio.append((pv_cur - pv_pre) / pv_pre)
            if save_value:
                if mode == 'test' and not rolling:
                    results_dict['test_pv'].append(state_test[2])
                    results_dict['test_pv_eq'].append(state_eq[2])
                    results_dict['test_action'].append(last_w)
                elif mode == 'val' and not rolling:
                    results_dict['val_pv'].append(state_test[2])
                    results_dict['val_pv_eq'].append(state_eq[2])
                    results_dict['val_action'].append(last_w)
                elif mode == 'test' and rolling:
                    results_dict['test_rolling_pv'].append(state_test[2])
                    results_dict['test_rolling_pv_eq'].append(state_eq[2])
                    results_dict['test_rolling_action'].append(last_w)
                elif mode == 'val' and rolling:
                    results_dict['val_rolling_pv'].append(state_test[2])
                    results_dict['val_rolling_pv_eq'].append(state_eq[2])
                    results_dict['val_rolling_action'].append(last_w)
                
            # save model
            if state_test[2] >= max_pv and rolling:
                torch.save(pi.state_dict(), '{}/ckpt/{}_rolling.pt'.format(base_dir, mode))
                max_pv = state_test[2]

            if rolling:
                train_eq_pv = np.array([1.]  *args.n_episode)
                for n_epi in range(num_rolling_steps):
                    # sample start index
                    i_starts = draw(start_index + i, size=args.n_episode)
                    #print('index: ', i_starts)
                    state, done = env.reset(init_w, 10000, t=i_starts)
                    last_w_roll = torch.from_numpy(state[1][:,1:]).to(device).float()
                    pi.optimizer.zero_grad()
                    for _ in range(args.episode_step):
                        x = torch.from_numpy(state[0]).to(device).float()
                        # take action
                        with torch.no_grad():
                            action = pi(x, last_w_roll)
                        state, reward, done = env.step(action.detach().cpu().numpy())
                        train_eq_pv *= reward.mean(1)
                        reward = torch.as_tensor(reward).to(device).float()
                        last_w_roll = torch.from_numpy(state[1][:,1:]).to(device).float()
                        L1 = pi.train_net(x, reward, last_w=last_w_roll, args=args)
                    pi.optimizer.step()

                    score= state[2].mean()
                    train_eq_score= train_eq_pv.mean()
                    if save_value:
                        if mode == 'test':
                            results_dict['test_rolling_rewards'].append(score)
                            results_dict['test_rolling_rewards_eq'].append(train_eq_score*10000)
                        if mode == 'val':
                            results_dict['val_rolling_rewards'].append(score)
                            results_dict['val_rolling_rewards_eq'].append(train_eq_score*10000)

                    print("-------------")
                    print(n_epi)
                    print("    ","training score:", score.item())
                    print("    ","training score_eq:", train_eq_score.item()*10000)

        # compute sharpe ratio
        sharpe_meam = np.mean(pv_ratio)
        sharpe_std = np.std(pv_ratio)
        sharpe_ratio = sharpe_meam / sharpe_std

        print('{}: equal final pv {}.'.format(mode, state_eq[2]))
        return state_test[2], sharpe_ratio

    score = 0
    max_val_score = 10000
    smooth_score = 10000
    train_eq_smooth_score = 1
    print_interval = 20
  
    # training
    for n_epi in range(num_steps):
        # sample start index
        i_starts = np.random.randint(low=window_size, high=train_steps - args.episode_step - 1, size=args.n_episode)
        state, done = env.reset(init_w, 10000, t=i_starts)
        last_w = torch.from_numpy(state[1][:,1:]).to(device).float()
        pi.optimizer.zero_grad()

        # step 4. train model
        train_eq_pv = np.array([1.]*args.n_episode)
        train_action = np.array([0.]*action_n)
        for i in range(episode_step):
            x = torch.from_numpy(state[0]).to(device).float()
            # take action
            with torch.no_grad():
                action = pi(x, last_w)
            state, reward, done = env.step(action.detach().cpu().numpy())

            train_eq_pv *= reward.mean(1)
            train_action += action.sum(0).cpu().numpy()
            reward = torch.as_tensor(reward).to(device).float()
            last_w = torch.from_numpy(state[1][:,1:]).to(device).float()

            L1 = pi.train_net(x, reward, last_w=last_w, args=args)
            results_dict['l1_losses'].append(L1)

        pi.optimizer.step()

        score= state[2].mean()
        smooth_score = 0.01*score.item() + 0.99 * smooth_score

        train_eq_score= train_eq_pv.mean()
        train_eq_smooth_score = 0.01*train_eq_score.item() + 0.99 * train_eq_smooth_score

        results_dict['train_rewards'].append(score)
        results_dict['train_smooth_rewards'].append(smooth_score)

        results_dict['train_rewards_eq'].append(train_eq_score*10000)
        results_dict['train_smooth_rewards_eq'].append(train_eq_smooth_score*10000)

        if n_epi%print_interval==0 and n_epi!=0:

            print("-------------")
            print(n_epi)
            print("    ","training score:", score.item())
            print("    ","training smooth score:", smooth_score)
            print( )
            print("    ","training score_eq:", train_eq_score.item()*10000)
            print("    ","training smooth score_eq:", train_eq_smooth_score*10000)
            print()
            print("    ","training action:", train_action)

            # step 5. save model
            val_score, _ = eval('val', n_epi)
            results_dict['val_scores'].append(val_score)
            if val_score >= max_val_score:
                torch.save(pi.state_dict(), '{}/ckpt/best.pt'.format(base_dir))
                max_val_score = val_score

    torch.save(pi.state_dict(), '{}/ckpt/last.pt'.format(base_dir))
    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)


    # step 6. evaluate
    # load best model
    pi.load_state_dict(torch.load('{}/ckpt/best.pt'.format(base_dir)))
    # without rolling
    val_score, val_sr = eval('val', rolling=False, save_value=True)
    test_score, test_sr = eval('test', rolling=False, save_value=True)
    # with rolling
    val_rolling_score, val_rolling_sr = eval('val', rolling=rolling, save_value=True)
    test_rolling_score, test_rolling_sr = eval('test', rolling=rolling, save_value=True)

    # compute val 40 pv and sharpe ratio
    val_steps = 40
    val_rolling_pv = np.array(results_dict['val_rolling_pv'])
    val_pv_list = val_rolling_pv[-val_steps-1:]
    init_pv = val_pv_list[0]
    val_pv_list = val_pv_list / init_pv * 10000
    pv_ratio = []
    for i in range(1, len(val_pv_list)):
        pv_ratio.append((val_pv_list[i] - val_pv_list[i-1]) / val_pv_list[i-1])
  
    sharpe_meam = np.mean(pv_ratio)
    sharpe_std = np.std(pv_ratio)
    val_40_sr = sharpe_meam / sharpe_std
    val_40_pv = val_pv_list[-1]

    # step 7. save results
    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)
    
    df = pd.read_csv('{}.csv'.format(args.file_name), encoding='utf8')
    df.loc[len(df)] = [exp_name,
                        max_val_score,
                        val_score,
                        val_sr,
                        test_score, 
                        test_sr, 
                        val_rolling_score,
                        val_rolling_sr,
                        test_rolling_score,
                        test_rolling_sr,
                        val_40_pv,
                        val_40_sr]
    df.to_csv('{}.csv'.format(args.file_name), encoding='utf8', index=False)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Execute Time: %s seconds." % (time.time() - start_time))
