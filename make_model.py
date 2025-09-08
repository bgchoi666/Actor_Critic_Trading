# Copyright 2021 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 주어진 모델 타입을 생성하여 주어진 구간에서 train, test, prediction

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import random
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math
import datetime
import random
import sys
import os

#from imblearn.under_sampling import *
#from imblearn.over_sampling import *
#from imblearn.combine import *

remove_columns = ['date']

input_size = 83
n_unit = 200
batch_size = 20
epochs = 30
train_size = 0.9
train_offset = 240
gubun = 2 # 0:predict only 1:test only 2:train
max_repeat_cnt = 100

pred_term = 5
norm_term = 20
target_type = 'C'
train_rate = 0.5
base1 = '고가'
base2 = '저가'
target_num = 3

reinfo_th = 0.3

last_train = '2022-06-30'
start_time = '2022/07/04/09:00'
end_time = '2022/07/15/15:00'


checkpoint_path = last_train+"/60M_input83_test"
checkpoint_path_best = last_train+"/60M_"+str(pred_term) + target_type + "_best"

model = ''

def create_model(mpdel_type):

    # 종가, 고가 기준에 따라 target_prob0, chkpoint_best file path 조정
    c = mpdel_type.find('C')
    h = mpdel_type.find('HL')
    p = mpdel_type.find('P')
    if c != -1:
        pred_term = int(mpdel_type[:c])
        target_type = 'C'
        base1 = '종가'
        base2 = '종가'
    elif h != -1:
        pred_term = int(mpdel_type[:h])
        target_type = 'HL'
        base1 = '고가'
        base2 = '저가'
    elif p != -1:
        pred_term = int(mpdel_type[:p])
        target_type = 'P'
        base1 = '종가'
        base2 = '종가'
    else:
        print("argument error " + mpdel_type)
        exit(0)

    if h != -1:
        checkpoint_path_best = last_train+"/" + "60M_" + str(pred_term) + "HL_best"
    elif c != -1:
        checkpoint_path_best = last_train+"/" + "60M_" + str(pred_term) + "C_best"
    elif p != -1:
        checkpoint_path_best = last_train+"/" + "60M_" + str(pred_term) + "P_best"
    else:
        print("argument error")
        exit(1)


    return pred_term, target_type, base1, base2, checkpoint_path_best

df0_path = 'kospi200f_11_60M.csv'
df_pred_path = 'kospi200f_60M_pred.csv'
result_path = 'pred_83_results.csv'


# 사실상 사용안함 , preprocessing은 'make_raw_data.py'에 의해 파생변수 생성하고 normalization은 수작업을 통해 *_pred,csv 생성
def preprocessing():
    # 필요 구간의 전처리 데이터 존재여부에 따라 처리

    if not os.path.isfile(df_pred_path):
        print("==============================================")
    else:
        norm_df0 = pd.read_csv(df_pred_path, encoding='euc-kr')

        df0 = pd.read_csv(df0_path, encoding='euc-kr')
        _start_time = df0.loc[df0['date'] >= start_time].min()['date']
        _end_time = df0.loc[df0['date'] <= end_time].max()['date']

        start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
        last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

        if last_date >= _end_time and start_date <= _start_time:
            print('nothing done! in this preprocessing')
            return

    df0 = pd.read_csv(df0_path, encoding='euc-kr')

    df0["시가대비종가변화율"] = (df0["종가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비고가변화율"] = (df0["고가"] - df0["시가"])/df0["시가"]*100
    df0["시가대비저가변화율"] = (df0["저가"] - df0["시가"])/df0["시가"]*100
    df0["종가대비고가변화율"] = (df0["고가"] - df0["종가"])/df0["종가"]*100
    df0["종가대비저가변화율"] = (df0["저가"] - df0["종가"])/df0["종가"]*100

    df0["1일전"] = np.concatenate([[0], df0["종가"].values[:-1]])
    df0["2일전"] = np.concatenate([[0, 0], df0["종가"].values[:-2]])
    df0["3일전"] = np.concatenate([[0, 0, 0], df0["종가"].values[:-3]])
    df0["4일전"] = np.concatenate([[0, 0, 0, 0], df0["종가"].values[:-4]])
    df0["5일전"] = np.concatenate([[0, 0, 0, 0, 0], df0["종가"].values[:-5]])

    df0["6일전"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["종가"].values[:-6]])
    df0["7일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-7]])
    df0["8일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-8]])
    df0["9일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-9]])
    df0["10일전"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["종가"].values[:-10]])

    df0["1일수익률"] = np.concatenate([np.zeros(1), df0["종가"].values[1:] - df0["종가"].values[:-1]])#rolling(window=2).apply(lambda x: x[1] - x[0])
    df0["3일수익률"] = np.concatenate([np.zeros(3), df0["종가"].values[3:] - df0["종가"].values[:-3]])#df0["종가"].rolling(window=4).apply(lambda x: x[3] - x[0])
    df0["5일수익률"] = np.concatenate([np.zeros(5), df0["종가"].values[5:] - df0["종가"].values[:-5]])#df0["종가"].rolling(window=6).apply(lambda x: x[5] - x[0])
    df0["10일수익률"] = np.concatenate([np.zeros(10), df0["종가"].values[10:] - df0["종가"].values[:-10]])#df0["종가"].rolling(window=11).apply(lambda x: x[10] - x[0])
    df0["20일수익률"] = np.concatenate([np.zeros(20), df0["종가"].values[20:] - df0["종가"].values[:-20]])#df0["종가"].rolling(window=21).apply(lambda x: x[20] - x[0])
    df0["40일수익률"] = np.concatenate([np.zeros(40), df0["종가"].values[40:] - df0["종가"].values[:-40]])#df0["종가"].rolling(window=41).apply(lambda x: x[40] - x[0])
    df0["60일수익률"] = np.concatenate([np.zeros(60), df0["종가"].values[60:] - df0["종가"].values[:-60]])#df0["종가"].rolling(window=61).apply(lambda x: x[60] - x[0])
    df0["90일수익률"] = np.concatenate([np.zeros(90), df0["종가"].values[90:] - df0["종가"].values[:-90]])#df0["종가"].rolling(window=91).apply(lambda x: x[90] - x[0])
    df0["120일수익률"] = np.concatenate([np.zeros(120), df0["종가"].values[120:] - df0["종가"].values[:-120]])#df0["종가"].rolling(window=121).apply(lambda x: x[120] - x[0])
    df0["180일수익률"] = np.concatenate([np.zeros(180), df0["종가"].values[180:] - df0["종가"].values[:-180]])#df0["종가"].rolling(window=181).apply(lambda x: x[180] - x[0])
    df0["240일수익률"] = np.concatenate([np.zeros(240), df0["종가"].values[240:] - df0["종가"].values[:-240]])#df0["종가"].rolling(window=241).apply(lambda x: x[240] - x[0])


    df0["5일최고"] = df0["고가"].rolling(window=5).max()
    df0["20일최고"] = df0["고가"].rolling(window=20).max()
    df0["60일최고"] = df0["고가"].rolling(window=60).max()
    df0["120일최고"] = df0["고가"].rolling(window=120).max()
    df0["240일최고"] = df0["고가"].rolling(window=240).max()

    df0["5일최저"] = df0["저가"].rolling(window=5).min()
    df0["20일최저"] = df0["저가"].rolling(window=20).min()
    df0["60일최저"] = df0["저가"].rolling(window=60).min()
    df0["120일최저"] = df0["저가"].rolling(window=120).min()
    df0["240일최저"] = df0["저가"].rolling(window=240).min()

    df0["1일전거래량"] = np.concatenate([[0], df0["거래량"].values[:-1]])
    df0["2일전거래량"] = np.concatenate([[0, 0], df0["거래량"].values[:-2]])
    df0["3일전거래량"] = np.concatenate([[0, 0, 0], df0["거래량"].values[:-3]])
    df0["4일전거래량"] = np.concatenate([[0, 0, 0, 0], df0["거래량"].values[:-4]])
    df0["5일전거래량"] = np.concatenate([[0, 0, 0, 0, 0], df0["거래량"].values[:-5]])

    df0["6일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0], df0["거래량"].values[:-6]])
    df0["7일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-7]])
    df0["8일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-8]])
    df0["9일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-9]])
    df0["10일전거래량"] = np.concatenate([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], df0["거래량"].values[:-10]])

    df0["1일거래변화량"] = np.concatenate([np.zeros(1), df0["거래량"].values[1:] - df0["거래량"].values[:-1]])#df0["거래량"].rolling(window=2).apply(lambda x: (x[1] - x[0])/x[0]*100)
    df0["3일거래변화량"] = np.concatenate([np.zeros(3), df0["거래량"].values[3:] - df0["거래량"].values[:-3]])#df0["거래량"].rolling(window=4).apply(lambda x: (x[3] - x[0])/x[0]*100)
    df0["5일거래변화량"] =  np.concatenate([np.zeros(5), df0["거래량"].values[5:] - df0["거래량"].values[:-5]])#df0["거래량"].rolling(window=6).apply(lambda x: (x[5] - x[0])/x[0]*100)
    df0["10일거래변화량"] = np.concatenate([np.zeros(10), df0["거래량"].values[10:] - df0["거래량"].values[:-10]])#df0["거래량"].rolling(window=11).apply(lambda x: (x[10] - x[0])/x[0]*100)
    df0["20일거래변화량"] = np.concatenate([np.zeros(20), df0["거래량"].values[20:] - df0["거래량"].values[:-20]])#df0["거래량"].rolling(window=21).apply(lambda x: (x[20] - x[0])/x[0]*100)
    df0["40일거래변화량"] = np.concatenate([np.zeros(40), df0["거래량"].values[40:] - df0["거래량"].values[:-40]])#df0["거래량"].rolling(window=41).apply(lambda x: (x[40] - x[0])/x[0]*100)
    df0["60일거래변화량"] = np.concatenate([np.zeros(60), df0["거래량"].values[60:] - df0["거래량"].values[:-60]])#df0["거래량"].rolling(window=61).apply(lambda x: (x[60] - x[0])/x[0]*100)
    df0["90일거래변화량"] = np.concatenate([np.zeros(90), df0["거래량"].values[90:] - df0["거래량"].values[:-90]])#df0["거래량"].rolling(window=91).apply(lambda x: (x[90] - x[0])/x[0]*100)
    df0["120일거래변화량"] = np.concatenate([np.zeros(120), df0["거래량"].values[120:] - df0["거래량"].values[:-120]])#df0["거래량"].rolling(window=121).apply(lambda x: (x[120] - x[0])/x[0]*100)
    df0["180일거래변화량"] = np.concatenate([np.zeros(180), df0["거래량"].values[180:] - df0["거래량"].values[:-180]])#df0["거래량"].rolling(window=181).apply(lambda x: (x[180] - x[0])/x[0]*100)
    df0["240일거래변화량"] = np.concatenate([np.zeros(240), df0["거래량"].values[240:] - df0["거래량"].values[:-240]])#df0["거래량"].rolling(window=241).apply(lambda x: (x[240] - x[0])/x[0]*100)

    df0["5일평균거래량"] = df0["거래량"].rolling(window=5).mean()
    df0["20일평균거래량"] = df0["거래량"].rolling(window=20).mean()
    df0["60일평균거래량"] = df0["거래량"].rolling(window=60).mean()
    df0["120일평균거래량"] = df0["거래량"].rolling(window=120).mean()
    df0["240일평균거래량"] = df0["거래량"].rolling(window=240).mean()

    df0["5일최고거래량"] = df0["거래량"].rolling(window=5).max()
    df0["20일최고거래량"] = df0["거래량"].rolling(window=20).max()
    df0["60일최고거래량"] = df0["거래량"].rolling(window=60).max()
    df0["120일최고거래량"] = df0["거래량"].rolling(window=120).max()
    df0["240일최고거래량"] = df0["거래량"].rolling(window=240).max()

    df0["5일최저거래량"] = df0["거래량"].rolling(window=5).min()
    df0["20일최저거래량"] = df0["거래량"].rolling(window=20).min()
    df0["60일최저거래량"] = df0["거래량"].rolling(window=60).min()
    df0["120일최저거래량"] = df0["거래량"].rolling(window=120).min()
    df0["240일최저거래량"] = df0["거래량"].rolling(window=240).min()

    start_index = df0.loc[df0['date'] >= start_time].index.min()
    end_index = df0.loc[df0['date'] <= end_time].index.max()

    df = df0[start_index - (norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, input_size+1):
            m = df.iloc[i:i+norm_term, j].mean()
            s = df.iloc[i:i+norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+norm_term-1, j] - m) / s
    norm_df.to_csv(df_pred_path, index=False, encoding='euc-kr')

    
def predict(model):
    df_pred = pd.read_csv(df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :input_size].reshape(-1, input_size)

    model.load_weights(checkpoint_path_best)
    pred = model.predict(pred_input)
    pred = np.argmax(pred, axis=1).reshape(-1)

    # 종가 검색
    df = pd.read_csv(df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= start_time].index.min()
    end_index = df.loc[df['date'] <= end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]

    #  0: 정상, 1: 급락 2:급등
    pred_results = []
    for i in range(len(dates)):
        pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
    pred_results = np.array(pred_results)

    pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(result_path, index=False, encoding='euc-kr')
    return pred_results[len(pred_results)-len(dates):]

if __name__ == "__main__":
    if len(sys.argv) > 4:
        model_type = sys.argv[1]
        last_train = sys.argv[2]
        start_time = sys.argv[3]
        end_time = sys.argv[4]

        if not os.path.isdir(last_train):
            os.makedirs(last_train)

        df_pred_path = last_train + '/kospi200f_60M_pred.csv'
        result_path = last_train + '/pred_83_results.csv'

    else:
        print('argument error!!')
        sys.exit(1)

    pred_term, target_type, base1, base2, checkpoint_path_best = create_model(model_type)

    print('data processing start...')
    now = datetime.datetime.now()
    print(now)
    preprocessing()
    print('data processing end...')

    r = predict(model)

    import profit
    profit.pred_term = pred_term
    profit.result_path = result_path
    p = profit.calc_profit()
    print(sys.argv[2] + " 수익률: " + str(p))

