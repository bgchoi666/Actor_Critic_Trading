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

    pred_term = 20

    # C: 종가 기준 평균값 비교, P: 시작, 종료 종가 비교, HL: 고가, 저가 평균 비교
    target_type = 'C'

    # target data 생성을 위한 가격 기준
    base1 = '종가'
    base2 = '종가'
    if target_type == 'HL':
        base1 = '고가'
        base2 = '저가'

    # 예측값 조정, 손절, 익절값
    reinfo_th = 0.0
    reinfo_width = 100
    loss_cut = 0.01
    profit_cut = 1

    input_size = 89
    n_unit = 200 # layer당 ubit 수
    norm_term = 20 # normalization을 위한 행 수

    # 시작시점, 종료시점, 학습모델이 있는 폴더
    start_time = '2023/08/16/09:00'
    end_time = '2023/08/17/09:00'
    last_train = '2023-08-15'

    model_pools = ["5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
              "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]


    # 원본 파일, 정규화 파일(학습용, 예측 용), 예측 결과 파일
    dir = "H:/알고리즘트레이딩4-1/"
    df0_path = dir + 'kospi200f_11_60M.csv'  # 원본 파일
    df_raw_path = dir + last_train + '/kospi200f_60M_raw.csv'  # 원본 파일에 target값 append
    norm_df_path = dir + last_train + '/kospi200f_60M_norm.csv'  # 학습용 normalization file
    df_pred_path = dir + last_train + '/kospi200f_60M_pred.csv'  # 예측용 normalization file
    result_path = dir + last_train + '/pred_88_results.csv'  # 예측 결과 손익 파일

    selected_num = 3
    selected_model_types = ['5C', '10HL', '15P']

    selected_checkpoint_path = ['', '', '']

    checkpoint_path = last_train + "/60M_input88_test"
    checkpoint_path_best = last_train+"/60M_"+str(pred_term) + target_type + "_best"

def set_path(conf, folder):
    if folder == 'H:/알고리즘5-1':
        conf.df0_path = folder + '/' + 'kospi200f_89_60M.csv'  # 원본 파일
    else:
        conf.df0_path = folder + '/' + 'kospi200f_11_60M.csv' # 원본 파일
    conf.df_raw_path = folder + '/' + conf.last_train+'/kospi200f_60M_raw.csv' # 원본 파일에 target값 append
    conf.norm_df_path = folder + '/' + conf.last_train+'/kospi200f_60M_norm.csv' # 학습용 normalization file
    conf.df_pred_path = folder + '/' + conf.last_train+'/kospi200f_60M_pred.csv' # 예측용 normalization file
    if folder == 'H:/알고리즘5-1':
        conf.result_path = folder + '/' + conf.last_train+'/pred_83_results.csv' # 예측 결과 손익 파일
    elif folder == 'H:/알고리즘트레이딩4-1':
        conf.result_path = folder + '/' + conf.last_train+'/pred_88_results.csv' # 예측 결과 손익 파일
    else:
        conf.result_path = folder + '/' + conf.last_train+'/pred_89_results.csv' # 예측 결과 손익 파일

def set_selected_checkpoint_path(conf, selected_model_types):
    conf.selected_model_types = selected_model_types
    conf.selected_num = len(selected_model_types)
    conf.selected_checkpoint_path = ['' for i in range(conf.selected_num)]
    for j in range(conf.selected_num):
        conf.selected_checkpoint_path[j] = conf.last_train + '/' + '60M_' + selected_model_types[j] + '_best'

def set_target_type(conf, target_type):
    # target data 생성을 위한 가격 기준
    conf.base1 = '종가'
    conf.base2 = '종가'
    if target_type == 'HL':
        conf.base1 = '고가'
        conf.base2 = '저가'

def set_pred_term(conf, pred_term):
    conf.pred_term = pred_term

def set_start_end_time(conf, start_time, end_time, last_train):
    conf.start_time = start_time
    conf.end_time = end_time
    conf.last_train = last_train

def set_ensemble(conf, selected_model_tpes):
    conf.selected_model_types = selected_model_tpes
    conf.selected_num = len(selected_model_tpes)
    conf.selected_checkpoint_path = ['' for i in range(conf.selected_num)]
    for j in range(conf.selected_num):
        conf.selected_checkpoint_path[j] = conf.dir + '/' + conf.last_train + '/' + '60M_' + conf.selected_model_types[j] + '_best'

def set_profit(conf, loss_cut, profit_cut):
    conf.loss_cut = loss_cut
    conf.profit_cut = profit_cut

def set_reinfo(conf, reinfo_th):
    conf.reinfo_th = reinfo_th

def set_width(conf, width):
    conf.reinfo_width = width   


def get_pre_last_train(last_train):
    if last_train[5:7] == '02':
        return last_train[:5] + '01-31'
    elif last_train[5:7] == '03':
        return last_train[:5] + '02-28'
    elif last_train[5:7] == '04':
        return last_train[:5] + '03-31'
    elif last_train[5:7] == '05':
        return last_train[:5] + '04-30'
    elif last_train[5:7] == '06':
        return last_train[:5] + '05-31'
    elif last_train[5:7] == '07':
        return last_train[:5] + '06-30'
    elif last_train[5:7] == '09':
        return last_train[:5] + '08-31'
    elif last_train[5:7] == '10':
        return last_train[:5] + '09-30'
    elif last_train[5:7] == '11':
        return last_train[:5] + '10-31'
    elif last_train[5:7] == '12':
        return last_train[:5] + '11-30'
    else:
        return (datetime.datetime.strptime(last_train, "%Y-%m-%d") - relativedelta(months=1)).strftime("%Y-%m-%d")

# *_pred.csv 파일이 존재할 떄 undersampling 비율 (target_prob*)에 따라 _norm.csv 파일 생성
def make_train_data(conf):
    df = pd.read_csv(conf.df_raw_path, encoding='euc-kr')
    norm_df = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term, 단순 예측 - 0: 하락 1: 상승
    if conf.target_type == 'P':
        target = []
        for i in range(len(df)):
            if i > len(df) - 1 - conf.pred_term:
                target.append(0)
            else:
                if df.loc[i, '종가'] > df.loc[i+conf.pred_term, '종가']*1.009:
                    target.append(1)
                elif df.loc[i, '종가'] < df.loc[i+conf.pred_term, '종가']*0.991:
                    target.append(2)
                else:
                    target.append(0)
        df['target'] = target
        norm_df['target'] = target[19:]
    else:
        target = []
        for i in range(len(df)):
            if i > len(df) - 1 - conf.pred_term:
                target.append(0)
            else:
                if df.loc[i, conf.base1] >= np.max(df.loc[i+1:i+conf.pred_term, conf.base1].values):
                    target.append(1)
                elif df.loc[i, conf.base2] <= np.min(df.loc[i+1:i+conf.pred_term, conf.base2].values):
                    target.append(2)
                else:
                    target.append(0)
        df['target'] = target
        norm_df['target'] = target[19:]

    df.to_csv(conf.df_raw_path, index=False, encoding='euc-kr')
    norm_df.to_csv(conf.norm_df_path, index=False, encoding='euc-kr')
    norm_df.to_csv(conf.df_pred_path, index=False, encoding='euc-kr')

    print('train을 위한 _norm.csv 파일 생성 완료')

# type 'P'에서 매수, 매도를 결정하는 임계값 upper, lower를 과거 해당 pred_term에 상응하는 n일 수익률 평균에 따라 설정
def make_train_data_new(conf):
    df = pd.read_csv(conf.df_raw_path, encoding='euc-kr')
    norm_df = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    # 고저점 예측 - 0: 정상  1: 고점 2: 저점 until pred_term, 단순 예측 - 0: 하락 1: 상승
    if conf.target_type == 'P':
        target = []
        for i in range(len(df)):

            if i < conf.pred_term:
                upper = 0
                lower = 0
            else:
                rates = np.array(df[:i]["종가"].rolling(window=conf.pred_term + 1).apply(lambda x: x[conf.pred_term] - x[0]))
                for i in range(conf.pred_term):
                    rates[i] = 0

                upper = rates[np.where(rates > 0)].mean() / conf.pred_term
                lower = rates[np.where(rates < 0)].mean() / conf.pred_term

            if i > len(df) - 1 - conf.pred_term:
                if df.loc[i, '종가'] + lower > df.loc[len(df) - 1, '종가']:
                    target.append(1)
                elif df.loc[i, '종가'] + upper < df.loc[len(df) - 1, '종가']:
                    target.append(2)
                else:
                    target.append(0)
            else:
                if df.loc[i, '종가'] + lower > df.loc[i+conf.pred_term, '종가']:
                    target.append(1)
                elif df.loc[i, '종가'] + upper < df.loc[i+conf.pred_term, '종가']:
                    target.append(2)
                else:
                    target.append(0)

        df['target'] = target
        norm_df['target'] = target[19:]

    else:
        target = []
        for i in range(len(df)):

            if i == len(df) - 1:
                target.append(0)

            elif i > len(df) - 1 - conf.pred_term:
                if df.loc[i, conf.base1] > np.max(df.loc[i + 1:len(df) - 1, conf.base1].values):
                    target.append(1)
                elif df.loc[i, conf.base2] < np.min(df.loc[i + 1:len(df) - 1, conf.base2].values):
                    target.append(2)
                else:
                    target.append(0)
            else:

                if df.loc[i, conf.base1] > np.max(df.loc[i+1:i+conf.pred_term, conf.base1].values):
                    target.append(1)
                elif df.loc[i, conf.base2] < np.min(df.loc[i+1:i+conf.pred_term, conf.base2].values):
                    target.append(2)
                else:
                    target.append(0)
        df['target'] = target
        norm_df['target'] = target[19:]

    df.to_csv(conf.df_raw_path, index=False, encoding='euc-kr')
    norm_df.to_csv(conf.norm_df_path, index=False, encoding='euc-kr')
    norm_df.to_csv(conf.df_pred_path, index=False, encoding='euc-kr')

    print('train을 위한 _norm.csv 파일 생성 완료')

# 사실상 사용안함 , preprocessing은 'make_raw_data.py'에 의해 파생변수 생성하고 normalization은 수작업을 통해 *_pred,csv 생성
def preprocessing(conf):
    # 필요 구간의 전처리 데이터 존재여부에 따라 처리

    if not os.path.isfile(conf.df_pred_path):
        print("==============================================")
    else:
        norm_df0 = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

        df0 = pd.read_csv(conf.df0_path, encoding='euc-kr')
        _end_time = df0.loc[df0['date'] <= conf.end_time].max()['date']
        if conf.start_time > df0['date'].values[-1]:
            conf.start_time = df0['date'].values[-1]
        _start_time = df0.loc[df0['date'] >= conf.start_time].min()['date']
        start_date = norm_df0.loc[norm_df0['date'].index.min(), 'date']
        last_date = norm_df0.loc[norm_df0['date'].index.max(), 'date']

        if conf.gubun == 0 or conf.gubun == 1: # for prediction
            if last_date >= _end_time and start_date <= _start_time:
                print('nothing done! in this preprocessing')
                return
        else:
            train_start_index = norm_df0.loc[norm_df0['date'] >= conf.start_time].index.min() - 1000
            if last_date >= _end_time and train_start_index > 0:
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

    df0["1일수익률"] = df0["종가"].rolling(window=2).apply(lambda x: x[1] - x[0], raw=True)
    df0["3일수익률"] = df0["종가"].rolling(window=4).apply(lambda x: x[3] - x[0], raw=True)
    df0["5일수익률"] = df0["종가"].rolling(window=6).apply(lambda x: x[5] - x[0], raw=True)
    df0["10일수익률"] = df0["종가"].rolling(window=11).apply(lambda x: x[10] - x[0], raw=True)
    df0["20일수익률"] = df0["종가"].rolling(window=21).apply(lambda x: x[20] - x[0], raw=True)
    df0["40일수익률"] = df0["종가"].rolling(window=41).apply(lambda x: x[40] - x[0], raw=True)
    df0["60일수익률"] = df0["종가"].rolling(window=61).apply(lambda x: x[60] - x[0], raw=True)
    df0["90일수익률"] = df0["종가"].rolling(window=91).apply(lambda x: x[90] - x[0], raw=True)
    df0["120일수익률"] = df0["종가"].rolling(window=121).apply(lambda x: x[120] - x[0], raw=True)
    df0["180일수익률"] = df0["종가"].rolling(window=181).apply(lambda x: x[180] - x[0], raw=True)
    df0["240일수익률"] = df0["종가"].rolling(window=241).apply(lambda x: x[240] - x[0], raw=True)

    if conf.dir != "H:알고리즘5-1":
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

    df0["1일거래변화량"] = df0["거래량"].rolling(window=2).apply(lambda x: x[1] - x[0], raw=True)
    df0["3일거래변화량"] = df0["거래량"].rolling(window=4).apply(lambda x: x[3] - x[0], raw=True)
    df0["5일거래변화량"] = df0["거래량"].rolling(window=6).apply(lambda x: x[5] - x[0], raw=True)
    df0["10일거래변화량"] = df0["거래량"].rolling(window=11).apply(lambda x: x[10] - x[0], raw=True)
    df0["20일거래변화량"] = df0["거래량"].rolling(window=21).apply(lambda x: x[20] - x[0], raw=True)
    df0["40일거래변화량"] = df0["거래량"].rolling(window=41).apply(lambda x: x[40] - x[0], raw=True)
    df0["60일거래변화량"] = df0["거래량"].rolling(window=61).apply(lambda x: x[60] - x[0], raw=True)
    df0["90일거래변화량"] = df0["거래량"].rolling(window=91).apply(lambda x: x[90] - x[0], raw=True)
    df0["120일거래변화량"] = df0["거래량"].rolling(window=121).apply(lambda x: x[120] - x[0], raw=True)
    df0["180일거래변화량"] = df0["거래량"].rolling(window=181).apply(lambda x: x[180] - x[0], raw=True)
    df0["240일거래변화량"] = df0["거래량"].rolling(window=241).apply(lambda x: x[240] - x[0], raw=True)

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

    if conf.gubun == 0:
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

    train_start_index = df0.loc[df0['date'] >= conf.start_time].index.min() - 1020
    if train_start_index < 19:
        train_start_index = 19
    train_end_index = df0.loc[df0['date'] <= conf.end_time].index.max()
    df = df0[train_start_index:train_end_index+1].reset_index(drop=True)
    df.to_csv(conf.df_raw_path, encoding='euc-kr')

    norm_df = df.copy()
    for i in range(conf.norm_term-1, len(norm_df)):
        for j in range(1, conf.input_size+1):
            m = df.iloc[i-(conf.norm_term-1):i+1, j].mean()
            s = df.iloc[i - (conf.norm_term-1):i+1, j].std()
            if s == 0:
                norm_df.iloc[i, j] = 0
            else:
                norm_df.iloc[i, j] = (df.iloc[i, j] - m) / s
    norm_df = norm_df.loc[conf.norm_term-1:].reset_index(drop=True)

    # save the normalized data for prediction
    norm_df.to_csv(conf.df_pred_path, index=False, encoding='euc-kr')