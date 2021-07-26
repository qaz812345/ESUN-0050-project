import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import font_manager

parser = argparse.ArgumentParser(description='Create training/test dataset')
parser.add_argument('--csv_file', type=str, help='The csv file path')
parser.add_argument('--exp_dir', type=str, help='The exp result dir path')
parser.add_argument('--train_start', type=str, help='Start date of training data', default='1995-01-01')
parser.add_argument('--train_end', type=str, help='End date of training data', default='2004-12-31')
parser.add_argument('--test_start', type=str, help='Start date of testing data', default='2005-01-01')
parser.add_argument('--test_end', type=str, help='End date of testing data', default='2008-12-31')
args = parser.parse_args()

csv_file = args.csv_file
exp_dir = args.exp_dir

df = pd.read_csv(csv_file, encoding='utf8')
dirs = {}
adrs = [exp_dir]
for adr in adrs:
    for dirPath, dirNames, fileNames in os.walk(adr):
        #print(dirPath)
        for f in fileNames:
            for exp in df['exp_name']:
                if exp in dirPath and 'results' in f:
                    dirs[os.path.join(dirPath, f)] = dirPath

# train data period
train_start_date = args.train_start
train_end_date = args.train_end
# test data period
test_start_date = args.test_start
test_end_date = args.test_end
# stock list
list_stock = ['1101', '1102', '1216', '1301', '1303', '1402', '2002', '2105', '2303', '2308']

# Get data date lists
dates = pd.read_csv('data/stock_{}_900101_210430.csv'.format(list_stock[0]), encoding='utf8').date
train_dates = dates[dates <= train_end_date]
train_dates = train_dates[train_dates >= train_start_date]
train_dates = train_dates.tolist()
test_dates = dates[dates <= test_end_date]
test_dates = test_dates[test_dates >= test_start_date]
test_dates = test_dates.tolist()
test_date_list = test_dates
test_date_list = pd.to_datetime(test_date_list)


colors = [i for i in get_cmap('tab20').colors]
plt.rcParams.update({'font.size': 24})
for f in dirs:
    print(f)
    fi = pd.read_pickle(f)

    # plot training rewards
    fig, axs = plt.subplots(2, figsize = (20, 10))
    axs[0].set_title('Train Rewards')
    axs[0].plot(fi['train_rewards'], label='reward')
    axs[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)

    axs[1].set_title('Train Smooth Rewards')
    axs[1].plot(fi['train_smooth_rewards'], label='agent')
    if 'train_smooth_rewards_eq' in fi:
        axs[1].plot(fi['train_smooth_rewards_eq'], label='equal')
    axs[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    fig.tight_layout()
    plt.savefig('{}/plot_train_reward.png'.format(dirs[f]))
    plt.close()


    # plot stock weight update curves
    val_steps = 300
    val_date_list = train_dates[-val_steps-1:]
    val_date_list = pd.to_datetime(val_date_list)

    fig, axs = plt.subplots(5, figsize = (30, 45))
    names = ['Money'] + list_stock
    w_list_eval = np.array(fi['val_action'])
    print('w_list_eval shape: ', w_list_eval.shape)
    axs[0].set_title('Portfolio Weights Update (val without rolling)')
    for j in range(len(names)):
        axs[0].plot(val_date_list[1:], w_list_eval[1:,j], color=colors[j], label='Weight Stock {}'.format(names[j]))
    axs[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    axs[0].tick_params(labelrotation=45)

    w_list_eval = np.array(fi['test_action'])
    axs[1].set_title('Portfolio Weights Update (test without rolling)')
    for j in range(len(names)):
        axs[1].plot(test_date_list[1:], w_list_eval[1:,j], color=colors[j], label='Weight Stock {}'.format(names[j]))
    axs[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    axs[1].tick_params(labelrotation=45)
  
    w_list_eval = np.array(fi['val_rolling_action'])
    axs[2].set_title('Portfolio Weights Update (val with rolling)')
    for j in range(len(names)):
        axs[2].plot(val_date_list[1:], w_list_eval[1:,j], color=colors[j], label='Weight Stock {}'.format(names[j]))
    axs[2].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    axs[2].tick_params(labelrotation=45)

    w_list_eval = np.array(fi['test_rolling_action'])
    axs[3].set_title('Portfolio Weights Update (test with rolling)')
    for j in range(len(names)):
        axs[3].plot(test_date_list[1:], w_list_eval[1:,j], color=colors[j], label='Weight Stock {}'.format(names[j]))
    axs[3].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    axs[3].tick_params(labelrotation=45)

    axs[4].set_title('Portfolio Money Weights Update (test with rolling)')
    axs[4].plot(test_date_list[1:], w_list_eval[1:,0], color=colors[0], label='Weight Stock {}'.format(names[0]))
    axs[4].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
    axs[4].tick_params(labelrotation=45)

    fig.tight_layout()
    plt.savefig('{}/plot_action.png'.format(dirs[f]))
    plt.close()


    # plot pv curves
    fig, axs = plt.subplots(4, figsize = (30, 40))
    if 'val_pv' in fi:
        axs[0].set_title('Validation Portfolio (without rolling)')
        axs[0].plot(val_date_list, fi['val_pv'], label='agent')
        axs[0].plot(val_date_list, fi['val_pv_eq'], label='equal')
        axs[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[0].tick_params(labelrotation=45)

    if 'test_pv' in fi:
        axs[1].set_title('Test Portfolio (without rolling)')
        axs[1].plot(test_date_list, fi['test_pv'], label='agent')
        axs[1].plot(test_date_list, fi['test_pv_eq'], label='equal')
        axs[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[1].tick_params(labelrotation=45)

    if 'val_rolling_pv' in fi:
        axs[2].set_title('Validation Portfolio (with rolling)')
        axs[2].plot(val_date_list, fi['val_rolling_pv'], label='agent')
        axs[2].plot(val_date_list, fi['val_rolling_pv_eq'], label='equal')
        axs[2].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[2].tick_params(labelrotation=45)

    if 'test_rolling_pv' in fi:
        axs[3].set_title('Test Portfolio (with rolling)')
        axs[3].plot(test_date_list, fi['test_rolling_pv'], label='agent')
        axs[3].plot(test_date_list, fi['test_rolling_pv_eq'], label='equal')
        axs[3].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[3].tick_params(labelrotation=45)
    
    fig.tight_layout()
    plt.savefig('{}/plot_pv.png'.format(dirs[f]))
    plt.close()


    #compute 40 date sr
    val_steps = 40
    val_date_list = train_dates[-val_steps-1:]
    val_date_list = pd.to_datetime(val_date_list)
    val_rolling_pv = np.array(fi['val_rolling_pv'])
    val_rolling_pv_eq = np.array(fi['val_rolling_pv_eq'])
    val_pv_list = val_rolling_pv[-val_steps-1:]
    val_pv_list_eq = val_rolling_pv_eq [-val_steps-1:]
    init_pv = val_pv_list[0]
    init_pv_eq = val_pv_list_eq[0]
    val_pv_list = val_pv_list / init_pv * 10000
    val_pv_list_eq = val_pv_list_eq / init_pv_eq * 10000

    fig, axs = plt.subplots(2, figsize = (30, 20))
    if 'val_rolling_pv' in fi:
        axs[0].set_title('Validation Portfolio (with rolling)')
        axs[0].plot(val_date_list, val_pv_list, label='agent')
        axs[0].plot(val_date_list, val_pv_list_eq, label='equal')
        axs[0].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[0].tick_params(labelrotation=45)

    if 'test_rolling_pv' in fi:
        axs[1].set_title('Test Portfolio (with rolling)')
        axs[1].plot(test_date_list, fi['test_rolling_pv'], label='agent')
        axs[1].plot(test_date_list, fi['test_rolling_pv_eq'], label='equal')
        axs[1].legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.5)
        axs[1].tick_params(labelrotation=45)
    fig.tight_layout()
    plt.savefig('{}/plot_val_{}_pv.png'.format(dirs[f], val_steps))
    plt.close()

