# Copyright 2023 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# 개별 모델의 creation, train, prediction

import data

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

import datetime
import random


import make_reinfo2 as mr


model = ''

def create_model(conf):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(conf.input_size)),
        tf.keras.layers.Dense(conf.n_unit, activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 2), activation='relu'),
        tf.keras.layers.Dense(int(conf.n_unit / 4), activation='relu'),
        tf.keras.layers.Dense(conf.target_num, activation='softmax')
    ])

    #cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #    checkpoint_path, verbose=1, save_weights_only=True,
        # 다섯 번째 에포크마다 가중치를 저장합니다
    #    save_freq=5)

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  #callbacks=[cp-callback]
                  metrics=['accuracy'])

    conf.checkpoint_path = conf.last_train+"/60M_input83_test"

    model.save_weights(conf.checkpoint_path)

    return model


def predict(conf):
    
    pred_model = create_model(conf)
    pred_model.load_weights(conf.last_train+"/60M_" + str(conf.pred_term) + conf.target_type + "_best")

    df_pred = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    start_index = df_pred.loc[df_pred['date'] >= conf.start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= conf.end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    pred_input = df_pred.values[start_index:end_index+1, :conf.input_size].reshape(-1, conf.input_size)

    #model.load_weights(checkpoint_path_best)
    pred = pred_model.predict(pred_input)
    pred = np.argmax(pred, axis=1).reshape(-1)

    # 종가 검색
    df = pd.read_csv(conf.df0_path, encoding='euc-kr')
    start_index = df.loc[df['date'] >= conf.start_time].index.min()
    end_index = df.loc[df['date'] <= conf.end_time].index.max()
    high = df['고가'].values[start_index:end_index + 1]
    low = df['저가'].values[start_index:end_index + 1]
    close = df['종가'].values[start_index:end_index + 1]
    open = df['시가'].values[start_index:end_index + 1]

    #  0: 고점, 1: 저점
    pred_results = [dates, pred, open, high, low, close]
    pred_results = np.array(pred_results).transpose()

    mr.th = conf.reinfo_th
    mr.pred_term = conf.pred_term
    mr.target_type = conf.target_type
    pred = mr.reinfo(pred, pred_results, conf.start_time, conf.reinfo_width)
    pred_results[:, 1] = pred

    # 앙상불 예측(ensemble.py)의 경우 개별 모델들의 result_path에 last_train 폴더를 붙이기 위해서.
    # 이는 자동 거래시 앙상블 예측의 결과가 끝나기 전에 모델의 결과를 앙상블의 결과로 읽는 오류를 막기 위해서이다.
    if conf.result_path.find(conf.last_train) < 0:
        result_path = conf.last_train + "/" + conf.result_path
    else:
        result_path = conf.result_path

    pd.DataFrame(np.array(pred_results), columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(result_path, index=False, encoding='euc-kr')
    return pred_results[len(pred_results)-len(dates):]

def parse(type):

    # 종가, 고가 기준에 따라 target_prob0, chkpoint_best file path 조정
    c = type.find('C')
    h = type.find('HL')
    p = type.find('P')
    if c != -1:
        pred_term = int(type[:c])
        target_type = 'C'
        base1 = '종가'
        base2 = '종가'
    elif h != -1:
        pred_term = int(type[:h])
        target_type = 'HL'
        base1 = '고가'
        base2 = '저가'
    elif p != -1:
        pred_term = int(type[:p])
        target_type = 'P'
        base1 = '종가'
        base2 = '종가'
    else:
        print("argument error " + type)
        exit(0)

    return pred_term, target_type, base1, base2
