# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Create training/test dataset')
parser.add_argument('--dataset', type=str, help='Dataset = [s10, s37]. (default: s10)', default='s10')
parser.add_argument('--start', type=str, help='Start date condition. (default: 1995-01-01)', default='1992-01-04')
parser.add_argument('--end', type=str, help='End date condition. (default: 2004-12-31)', default='2021-04-10')
parser.add_argument('--output', type=str, help='Output file name. Default: data_start_end.npy ', default='default')
args = parser.parse_args()

start_date = args.start 
end_date = args.end
dataset = args.dataset
output_name = args.output


if dataset == 's10':
    stock_ids = ['1101', '1102', '1216', '1301', '1303', '1402', '2002', '2105', '2303', '2308']
    stock_dict = {'1101':'台泥', '1102':'亞泥', '1216':'統一', '1301':'台塑', '1303':'南亞',
                '1402':'遠東新', '2002':'中鋼', '2105':'正新', '2303':'聯電', '2308':'台達電'}
elif dataset == 's37':
    stock_ids = ['1101', '1102', '1216', '1227', '1229', '1301', '1303', '1402', '1434', '1504',
                '1605', '1717', '1718', '1802', '1907', '2002', '2105', '2201', '2308', '2312',
                '2313', '2317', '2324', '2327', '2330', '2603', '2606', '2609', '2610', '2801',
                '2809', '2903', '9904', '9910', '9914', '9917', '9921']
    stock_dict = {'1101':'台泥', '1102':'亞泥', '1216':'統一', '1227':'佳格', '1229':'聯華投控',
                '1301':'台塑', '1303':'南亞', '1402':'遠東新', '1434':'福懋', '1504':'東元',
                '1605':'華新', '1717':'長興', '1718':'中纖', '1802':'台玻', '1907':'永豐餘',
                '2002':'中鋼', '2105':'正新', '2201':'裕隆', '2308':'台達電', '2312':'金寶',
                '2313':'華通', '2317':'鴻海', '2324':'仁寶', '2327':'國巨', '2330':'台積電',
                '2603':'長榮', '2606':'裕民', '2609':'陽明', '2610':'華航', '2801':'彰銀',
                '2809':'京城銀', '2903':'遠百', '9904':'寶成', '9910':'豐泰', '9914':'美利達',
                '9917':'中保科', '9921':'巨大'}

# read csv files
list_open = list()
list_close = list()
list_high = list()
list_low = list()
list_volume = list()

list_df = list()
list_date = list()

for i, s in enumerate(stock_ids):
    data = pd.read_csv('data/stock_{}_900101_210430.csv'.format(s), encoding='utf8')
    data = data[data['date'] <= end_date]
    data = data[data['date'] >= start_date]
    list_df.append(data)
    if i == 0:
        list_date = data['date'].tolist()

for df in list_df: 
    opens = list()
    close = list()
    high = list()
    low = list()
    volume = list()
    for d in list_date:
        if d in df['date'].tolist():
            df.set_index("date" , inplace=True)
            opens.append(df.loc[d, 'open'])
            close.append(df.loc[d, 'close'])
            high.append(df.loc[d, 'max'])
            low.append(df.loc[d, 'min'])
            volume.append(df.loc[d, 'Trading_Volume'])
            df.reset_index(inplace=True)
        else:
            # missing value
            print('no data in {}'.format(d))
            opens.append(opens[-1])
            close.append(close[-1])
            high.append(high[-1])
            low.append(low[-1])
            volume.append(0)
        
    list_open.append(opens)
    list_close.append(close)
    list_high.append(high)
    list_low.append(low)
    list_volume.append(volume)

array_open = np.array(list_open)[:, 1:]
array_open_yesterday = np.array(list_open)[:, :-1]
array_close_yesterday = np.array(list_close)[:, :-1]
array_high_yesterday = np.array(list_high)[:, :-1]
array_low_yesterday = np.array(list_low)[:, :-1]
min_volume = np.min(np.array(list_volume))
max_volume = np.max(np.array(list_volume))
print('Min volume: ', min_volume)
print('Max volume: ', max_volume)
print('Mean volume: ', np.mean(np.array(list_volume)))
array_volume_yesterday = np.array(list_volume)[:, :-1]

print('open shaep: ', array_open_yesterday.shape)
print('close shaep: ', array_close_yesterday.shape)
print('high shaep: ', array_high_yesterday.shape)
print('low shaep: ', array_low_yesterday.shape)
print('volume shaep: ', array_volume_yesterday.shape)

X = np.array([(array_volume_yesterday - min_volume)/(max_volume - min_volume),
            array_close_yesterday/array_open_yesterday, 
            array_high_yesterday/array_open_yesterday,
            array_low_yesterday/array_open_yesterday,
            array_open/array_open_yesterday])
          
print('X shape: ', X.shape)
np.save('./np_data/{}.npy'.format(output_name), X)