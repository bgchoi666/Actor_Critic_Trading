# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# configuration 및 data 전처리 process

import pandas as pd
import numpy as np
import os
import datetime
from dateutil.relativedelta import relativedelta

class config:

    gubun = 2 # 0:predict only 1:test only 2:train

    # training parameters
    batch_size = 20
    epochs=30
    train_size = 0.9
    train_offset = 240
    train_rate = 0.5
    target_num = 3
    max_repeat_cnt = 100
    trading_9h = False

    pred_term = 10

    # C: 종가 기준 평균값 비교, P: 시작, 종료 종가 비교, HL: 고가, 저가 평균 비교
    target_type = 'C'

    # target data 생성을 위한 가격 기준
    base1 = '종가'
    base2 = '종가'
    if target_type == 'HL':
        base1 = '고가'
        base2 = '저가'

    # 예측값 조정, 손절, 익절값
    reinfo_th = 0.4
    reinfo_width = 70
    loss_cut = 0.01
    profit_cut = 1

    input_size = 83
    n_unit = 200 # layer당 ubit 수
    norm_term = 20 # normalization을 위한 행 수

    # 시작시점, 종료시점, 학습모델이 있는 폴더
    start_time = '2022/01/01/09:00'
    end_time = '2022/01/31/15:00'
    last_train = '2021-12-31'

    model_pools = ["5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
              "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]


    # 원본 파일, 정규화 파일(학습용, 예측 용), 예측 결과 파일
    df0_path = 'kospi200f_60M.csv'  # 원본 파일
    df_pred_path = 'kospi200f_60M_pred.csv'  # 예측용 normalization file
    result_path = 'pred_83_results.csv'  # 예측 결과 손익 파일

    selected_num = 3
    selected_model_types = ['5C', '10HL', '15P']


    
# 사실상 사용안함 , preprocessing은 'make_raw_data.py'에 의해 파생변수 생성하고 normalization은 수작업을 통해 *_pred,csv 생성
def preprocessing(conf):
    # 필요 구간의 전처리 데이터 존재여부에 따라 처리

    if not os.path.isfile(conf.df_pred_path):
        print("==============================================")
    else:
        norm_df0 = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

        df0 = pd.read_csv(conf.df0_path, encoding='euc-kr')
        _end_time = df0.loc[df0['date'] <= conf.end_time].max()['date']
        _start_time = df0.loc[df0['date'] >= conf.start_time].min()['date']
        start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
        last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

        if last_date >= _end_time and start_date <= _start_time:
            print('nothing done! in this preprocessing')
            return    	

    df0 = pd.read_csv(conf.df0_path, encoding='euc-kr')

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

    df0["1일수익률"] = np.concatenate([np.zeros(1), df0["종가"].values[1:] - df0["종가"].values[:-1]])
    df0["3일수익률"] = np.concatenate([np.zeros(3), df0["종가"].values[3:] - df0["종가"].values[:-3]])
    df0["5일수익률"] = np.concatenate([np.zeros(5), df0["종가"].values[5:] - df0["종가"].values[:-5]])
    df0["10일수익률"] = np.concatenate([np.zeros(10), df0["종가"].values[10:] - df0["종가"].values[:-10]])
    df0["20일수익률"] = np.concatenate([np.zeros(20), df0["종가"].values[20:] - df0["종가"].values[:-20]])
    df0["40일수익률"] = np.concatenate([np.zeros(40), df0["종가"].values[40:] - df0["종가"].values[:-40]])
    df0["60일수익률"] = np.concatenate([np.zeros(60), df0["종가"].values[60:] - df0["종가"].values[:-60]])
    df0["90일수익률"] = np.concatenate([np.zeros(90), df0["종가"].values[90:] - df0["종가"].values[:-90]])
    df0["120일수익률"] = np.concatenate([np.zeros(120), df0["종가"].values[120:] - df0["종가"].values[:-120]])
    df0["180일수익률"] = np.concatenate([np.zeros(180), df0["종가"].values[180:] - df0["종가"].values[:-180]])
    df0["240일수익률"] = np.concatenate([np.zeros(240), df0["종가"].values[240:] - df0["종가"].values[:-240]])

    # 알고리즘4-1일때만 사용 알고리즘2-1에서 삭제
    df0["5일평균"] = df0["종가"].rolling(window=5).mean()
    df0["20일평균"] = df0["종가"].rolling(window=20).mean()
    df0["60일평균"] = df0["종가"].rolling(window=60).mean()
    df0["120일평균"] = df0["종가"].rolling(window=120).mean()
    df0["240일평균"] = df0["종가"].rolling(window=240).mean()
    
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

    df0["1일거래변화량"] = np.concatenate([np.zeros(1), df0["거래량"].values[1:] - df0["거래량"].values[:-1]])
    df0["3일거래변화량"] = np.concatenate([np.zeros(3), df0["거래량"].values[3:] - df0["거래량"].values[:-3]])
    df0["5일거래변화량"] =  np.concatenate([np.zeros(5), df0["거래량"].values[5:] - df0["거래량"].values[:-5]])
    df0["10일거래변화량"] = np.concatenate([np.zeros(10), df0["거래량"].values[10:] - df0["거래량"].values[:-10]])
    df0["20일거래변화량"] = np.concatenate([np.zeros(20), df0["거래량"].values[20:] - df0["거래량"].values[:-20]])
    df0["40일거래변화량"] = np.concatenate([np.zeros(40), df0["거래량"].values[40:] - df0["거래량"].values[:-40]])
    df0["60일거래변화량"] = np.concatenate([np.zeros(60), df0["거래량"].values[60:] - df0["거래량"].values[:-60]])
    df0["90일거래변화량"] = np.concatenate([np.zeros(90), df0["거래량"].values[90:] - df0["거래량"].values[:-90]])
    df0["120일거래변화량"] = np.concatenate([np.zeros(120), df0["거래량"].values[120:] - df0["거래량"].values[:-120]])
    df0["180일거래변화량"] = np.concatenate([np.zeros(180), df0["거래량"].values[180:] - df0["거래량"].values[:-180]])
    df0["240일거래변화량"] = np.concatenate([np.zeros(240), df0["거래량"].values[240:] - df0["거래량"].values[:-240]])
    
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

    start_index = df0.loc[df0['date'] >= conf.start_time].index.min()
    end_index = df0.loc[df0['date'] <= conf.end_time].index.max()

    df = df0[start_index - (conf.norm_term-1) : end_index + 1].reset_index(drop=True)
    norm_df = df[conf.norm_term-1 : ].reset_index(drop=True).copy()

    for i in range(end_index - start_index + 1):
        for j in range(1, conf.input_size+1):
            m = df.iloc[i:i+conf.norm_term, j].mean()
            s = df.iloc[i:i+conf.norm_term, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i+conf.norm_term-1, j] - m) / s
    norm_df.to_csv(conf.df_pred_path, index=False, encoding='euc-kr')
    return
