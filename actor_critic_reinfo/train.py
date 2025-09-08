# Copyright 2025 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# Advatage Actor-Critic 최적의 reinfo 값을 출력하는 모델을 훈련시킨다.
# state는 알고리즘2-1, 4-1, 5-1의 데이터이고 reward를 제공하는 환경은 주어진 알고리즘 범위내에서
# random으로 선택된 앙상블 모델의 봉과 봉사이의 reinfo action에 따른 수익률이고 (expectied return,
# critic 모델의 value), (actor 모델의 action probability) 등으로 weight이 update된다.


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

import collections
import statistics
import tqdm
import random
import datetime

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

import data
from data import config as conf
import ensemble_proc as ep
import make_reinfo2 as mr
import profit

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

conf.reinfo_th = 0.1
conf.loss_cut = 0.005

conf.gubun = 0
# 앙상블내의 모델 개수 설정
conf.selected_num = 3

ep.mr = mr
ep.profit.slippage = 0.05
ep.margin = 10000000
ep.profit.margin = 10000000

eps = 0.000001

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_outputs,
            num_hidden_units):
        """Initialize."""
        super().__init__()

        self.common1 = keras.layers.Dense(num_hidden_units, activation="relu")
        self.common2 = keras.layers.Dense(int(num_hidden_units / 2), activation="relu")
        self.actor = keras.layers.Dense(num_outputs, activation="softmax")
        self.critic = keras.layers.Dense(1)

    def call(self, inputs):
        x1 = self.common1(inputs)
        x2 = self.common2(x1)
        return self.actor(x2), self.critic(x2)

# A2C 모델 초기화
actor_critic = ActorCritic(5, 128)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def compute_loss(
        action_probs,
        values,
        returns):
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

def get_expected_return(rewards,gamma,standardize=True):
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

    return returns

def pred_input(conf):
    df_pred = pd.read_csv(conf.df_pred_path, encoding='euc-kr')

    if conf.start_time > df_pred['date'].values[-1]:
        conf.start_time = df_pred['date'].values[-1]

    start_index = df_pred.loc[df_pred['date'] >= conf.start_time].index.min()
    end_index = df_pred.loc[df_pred['date'] <= conf.end_time].index.max()

    dates = df_pred.pop('date').values[start_index:end_index + 1].reshape(-1)

    df_pred = df_pred.astype(float)

    pred_input = df_pred.values[start_index:end_index+1, :conf.input_size].reshape(-1, conf.input_size)

    return pred_input

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#@tf.function
def train_step(conf, actor_critic, input):
    """
    conf: 모든 정보
    model: actor-critic model
    """

    with tf.GradientTape() as tape:
        # actor-critic model에서 reinfo action과 value 출력
        action_probs_t, values = actor_critic(input)
        values = tf.reshape(values, [-1])
        actions = tf.argmax(action_probs_t, 1)

        # ep.predict에서 각 봉별 수익률에 의한 rewards 출력
        rewards = create_rewards(actions, conf)

        action_probs = []
        for k in range(len(actions)):
            action_probs.append(action_probs_t[k, actions[k]])

        action_probs = np.array(action_probs)

        # expected_returns 계산
        returns = get_expected_return(rewards, 0.99)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculate the loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # gradient, optimizer 실행
    grads = tape.gradient(loss, actor_critic.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, actor_critic.trainable_variables))

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward

def create_rewards(actions, conf):

    pred_results = pd.read_csv('test_results.csv', encoding='euc-kr').values[:, :6]
    pred_before = pred_results[:, 1]
    close = pred_results[:, 5]
    mr.ths = actions

    mr.pred_term = conf.pred_term
    mr.target_type = conf.target_type
    pred = mr.reinfo_flex(pred_before, pred_results, conf.start_time, conf.reinfo_width)
    pred_results[:, 1] = pred

    rewards = []
    state = 0
    for i in range(len(pred)-1):
        if pred[i] == 1:
            rewards.append((1 - close[i+1] / close[i])*100)
            state = 1
        elif pred[i] == 2:
            rewards.append((close[i+1] / close[i] - 1)*100)
            state = 2
        elif state == 1:
            rewards.append((1 - close[i+1] / close[i])*100)
        elif state == 2:
            rewards.append((close[i + 1] / close[i] - 1) * 100)
        else:
            rewards.append(0)
    rewards.append(0)

    return np.array(rewards, dtype=float)

def train(conf, model_name, rwd_th, max_eps, min_eps):

    # ensemble_proc에 있는 start, end times list값을 conf.start, end time과 일치시킴
    ep.start_times[np.where(np.array(ep.start_times) <= conf.start_time)[0][-1]] = conf.start_time
    ep.end_times[np.where(np.array(ep.end_times) >= conf.end_time)[0][0]] = conf.end_time

    #data.set_path(conf, dir)
    ep.last_trains, ep.start_times, ep.end_times, ep.term = ep.set_term(conf, ep.last_trains, ep.start_times,
                                                                        ep.end_times, 'reinfo_train')
    min_episodes_criterion = min_eps
    max_episodes = max_eps

    # `CartPole-v1` is considered solved if average reward is >= 475 over 500
    # consecutive trials
    reward_threshold = rwd_th
    running_reward = 0

    # The discount factor for future rewards
    gamma = 0.99

    # Keep the last episodes reward
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

    t = tqdm.trange(max_episodes)
    ps = []
    for i in t:
        # 15일 단위로 random 선택
        idx = random.randrange(0, len(ep.last_trains) - 1)
        conf.last_train = ep.last_trains[idx]
        conf.start_time = ep.start_times[idx]
        conf.end_time = ep.end_times[idx]

        # random으로 선택된 앙상블과 pred_term, 전체 구간중에 한 구간(15일)을 state로 하여 주어진 구간에서 A2C 모델을 학습
        #data.set_ensemble(conf, sorted(random.sample(conf.model_pools, conf.selected_num)))
        #conf.reinfo_th = random.randrange(0, 101, 5) / 100
        #conf.pred_term = random.sample([1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40], 1)[0]
        #conf.reinfo_width = random.randrange(5, 71, 5)
        #conf.loss_cut =random.randrange(0, 51, 5) / 1000

        # actor-critic input data 생성
        data.set_path(conf, conf.dir)
        data.preprocessing(conf)
        input = pred_input(conf)

        # reinfo를 적용하지 않은 예측 결과 파일 획득
        conf.result_path = 'test_results.csv'
        p = ep.predict(conf)
        if len(ps) > 100:
           ps = list(np.array(ps)[1:])
        ps.append(p)
        print("reinfo 적용전 평균 수익률: ", str(np.array(ps).mean()))

        episode_reward = train_step(conf, actor_critic, input)

        episodes_reward.append(episode_reward.numpy())
        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
            pass  # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

    actor_critic.save(model_name)

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

def test(conf, model_name):
    actor_critic = tf.keras.models.load_model(model_name)

    data.set_path(conf, conf.dir)

    # actor-critic model에서 reinfo action과 value 출력
    data.preprocessing(conf)
    input = pred_input(conf)
    action_probs_t, values = actor_critic(input)
    actions = tf.argmax(action_probs_t, 1)

    # 적용전 예측 출력
    conf.result_path = 'test_results.csv'
    p = ep.predict(conf)
    pred_results = pd.read_csv('test_results.csv', encoding='euc-kr').values[:, :6]
    pred = pred_results[:, 1]
    print("적용전 수익률: ", str(p))

    # action을 make_reinfo2에 전달
    ep.mr.ths = actions
    ep.mr.pred_term = conf.pred_term
    ep.mr.target_type = conf.target_type
    pred = ep.mr.reinfo_flex(pred, pred_results, conf.start_time, conf.reinfo_width)
    pred_results[:, 1] = np.array(pred)

    # 결과 파일에 저장
    # 0: 정상, 1: 고점 2:저점
    pd.DataFrame(pred_results, columns=['date', 'result', 'open', 'high', 'low', 'close']).to_csv(conf.result_path, index=False, encoding='euc-kr')

    # 수익률 계산하여 return
    profit.loss_cut = conf.loss_cut
    profit.pred_term = conf.pred_term
    profit.result_path = conf.result_path

    p = profit.calc_profit()

    term = datetime.datetime.strptime(conf.start_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d") + "~" + \
           datetime.datetime.strptime(conf.end_time, "%Y/%m/%d/%H:%M").strftime("%Y-%m-%d")

    print(term + " 수익률: ", str(p))

if __name__ == "__main__":
    mode = 'train-'

    conf.input_size = 83
    conf.dir = "H:/알고리즘트레이딩2-1"
    model_name = '알고리즘2-1/2022-01-01~2025-01-31_10C_30C_5C_0.3_10_25_0.001'

    rwd_th = 2.5
    max_eps = 1000
    min_eps = 20

    trained_model = ["10C", "30C", "5C", 0.3, 10, 25, 0.001]
    conf.reinfo_th = trained_model[3]
    conf.pred_term = trained_model[4]
    conf.reinfo_width = trained_model[5]
    conf.loss_cut = trained_model[6]

    if mode == 'train':
        conf.start_time = '2022/01/01/09:00'
        conf.end_time = '2025/01/31/15:00'
        conf.last_train = '2025-01-31'

        data.set_ensemble(conf, trained_model[:3])
        train(conf, model_name, rwd_th, max_eps, min_eps)
    else:
        conf.start_time = '2025/02/01/09:00'
        conf.end_time = '2025/02/20/15:00'
        conf.last_train = '2025-01-31'

        data.set_ensemble(conf, trained_model[:3])
        test(conf, model_name)

    exit(0)