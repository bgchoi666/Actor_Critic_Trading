# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 앙상블의 기간별 수익률 조사와 관련된 모듈들

import data
from data import config as conf
import model as md
import make_reinfo2 as mr
import profit

import pandas as pd
import numpy as np
from datetime import datetime
import os
import random

from tensorflow import keras

term = ''

folder = "앙상블실험"

result_path = "eval_reflection/eval_reflection_" + term + "_losscut" + str(conf.loss_cut) + ".csv"

hit_ratios = [0.2, 0.3, 0.4, 0.5]
eval_terms = [10, 20, 30, 40]
eval_widths = [10, 20, 30, 40, 50, 60, 70]

every_term_random = False
bayesian = False


# 빈 데이터프레임 생성
model_pools = ["5C", "5HL", "5P", "10C", "10HL", "10P", "15C", "15HL", "15P", "20C", "20HL", "20P",
          "25C", "25HL", "25P", "30C", "30HL", "30P", "40C", "40HL", "40P"]

model_path = ""

def set_ensemble(conf, selected_model_tpes):
    conf.selected_model_types = selected_model_tpes
    conf.selected_num = len(selected_model_tpes)
    conf.selected_checkpoint_path = ['' for i in range(conf.selected_num)]
    for j in range(conf.selected_num):
        conf.selected_checkpoint_path[j] = conf.last_train + '/' + '60M_' + conf.selected_model_types[j] + '_best'


def predict(conf, pos, last_time, num):
    # create prediction values
    df_pred = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= conf.start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= conf.end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    #개별 모델 예측 전 앙상블 pred_term, target_type save하고 개별 모델 예측후 복원
    save_term = conf.pred_term
    save_type = conf.target_type

    #모델들의 예측값 생성
    pred_model = md.create_model(conf)
    r = []
    for i in range(conf.selected_num):
        
        model_pred_term, model_target_type = parsing(conf.selected_model_types[i])
        conf.pred_term = model_pred_term
        conf.target_type = model_target_type

        pred_model.load_weights(conf.last_train + '/' + '60M_' + conf.selected_model_types[i] + '_best')
        
        pred = md.predict(conf)[:, 1]

        r.append(pred)

    r = np.array(r)

    conf.pred_term = save_term
    conf.target_type = save_type

    #앙상블의 예측값 생성
    pred = []
    for i in range(len(r[0])):
        cnt = [0, 0, 0]
        for j in range(conf.selected_num):
            cnt[r[j][i]] += 1
        pred.append(np.argmax(cnt))

    # 시가, 고가, 저가, 종가 검색
    df = pd.read_csv(conf.df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= conf.start_time].index.min()
    end_index = df.loc[df['date'] <= conf.end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]
    #  0: 고점, 1: 저점

    pred_results = []
    for i in range(len(pred)):
        pred_results.append([dates[i], pred[i], open[i], high[i], low[i], close[i]])
    pred_results = np.array(pred_results)

    mr.th = conf.reinfo_th
    mr.pred_term = conf.pred_term
    mr.target_type = conf.target_type
    pred = mr.reinfo(pred, pred_results, conf.start_time, conf.reinfo_width)
    pred_results[:, 1] = np.array(pred)


    # 결과 파일에 저장
    # 0: 정상, 1: 고점 2:저점
    pd.DataFrame(pred_results, columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(conf.result_path, index=False, encoding='euc-kr')

    # 평가손을 계산하여 return
    idx = pred_results.indexOf(last_time, 0)
    start_close = pred_results[idx, 5]
    end_close = pred_results[-1, 5]
    signal = pred_results[-1, 1]
    
    if pos == signal

def parsing(model_type):

    c = model_type.find('C')
    h = model_type.find('HL')
    p = model_type.find('P')
    if c != -1:
        pred_term = int(model_type[:c])
        target_type = 'C'
    elif h != -1:
        pred_term = int(model_type[:h])
        target_type = 'HL'
    elif p != -1:
        pred_term = int(model_type[:p])
        target_type = 'P'
    else:
        print("argument error " + model_type)
        exit(0)

    return pred_term, target_type

