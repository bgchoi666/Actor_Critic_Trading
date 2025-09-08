# Copyright 2025 Bimghi Choi. All Rights Reserved.

# -*- coding:utf-8 -*-

# Advatage Actor-Critic 최적의 reinfo 값을 출력하는 모델을 훈련시킨다.
# state는 알고리즘2-1, 4-1, 5-1의 이전 100봉의 수익률 추이이고 reinfo action에 따라 변경되는 수익률이 new state이고
# 이것으로부터 reward와 expected return이 계산된다.


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
            num_outputs: int,
            num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common1 = keras.layers.Dense(num_hidden_units, activation="relu")
        self.common2 = keras.layers.Dense(int(num_hidden_units / 2), activation="relu")
        self.actor = keras.layers.Dense(num_outputs, activation="softmax")
        self.critic = keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x1 = self.common1(inputs)
        x2 = self.common2(x1)
        return self.actor(x2), self.critic(x2)

# A2C 모델 초기화
actor_critic = ActorCritic(20, 128)

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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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

def state_rate_input(conf):
    # 수익률 결과 파일 생성후 시그널 state와 봉간 수익률 추이를 input으로 append

    data.preprocessing(conf)

    # 전체 수익 파일 생성
    conf.result_path = "test_results.csv"
    ep.predict(conf)
    results_df = pd.read_csv(conf.result_path, encoding='euc-kr')

    # 봉간 rate과 보유 상태 수집
    inputs = []
    pred = results_df['result'].values
    close = results_df['close'].values
    date = results_df['date'].values
    state = 0
    rate = 0
    for i in range(len(results_df)):
        inputs.append([state, rate])

        if i == len(results_df) - 1:
            break

        # state값 변경
        if date[i][11:16] >= '15:00':
            state = 0
        elif state == 0:
            state = pred[i]
        elif state != pred[i] and pred[i] != 0:
            state = pred[i]

        # 이전 봉의 예측값이 state, rate은 현재 봉과 이전 봉의 종가 상승/하락률
        if state == 0:
            rate = 0
        elif state == 1:
            rate = (1 - close[i+1]/close[i])*100
        else:
            rate = (close[i+1] / close[i] - 1) * 100

    return np.array(inputs), results_df.values

class env:
    """Combined actor-critic network."""

    def __init__(self, conf):

        """Initialize."""
        self.conf = conf
        self.df = pd.read_csv(conf.df0_path, encoding='euc-kr')
        self.state = tf.constant(np.zeros(conf.input_size+2), dtype=tf.float32)
        self.cnt = 0
        self.max_cnt = 0

    def step(self, action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        self.cnt += 1

        if self.cnt > self.max_cnt - 1:
            return self.state, tf.constant(0)

        mr.th = action / tf.constant(20)
        mr.pred_term = self.conf.pred_term
        mr.target_type = self.conf.target_type
        pred = mr.reinfo(self.results[:self.cnt, 1], self.results[:self.cnt, :6], conf.start_time,
                         self.conf.reinfo_width)
        self.results[:self.cnt, 1] = np.array(pred)

        close = self.results[:self.cnt+1, 5]

        # 변경된 pred로 인한 보유 상태값 save
        signal_state = 0
        for i in range(self.cnt-1, -1, -1):
            if pred[i] == 1 or pred[i] == 2:
                signal_state = pred[i]
                break
            else:
                continue

        if signal_state == 1:
            reward = (1 - close[self.cnt] / close[self.cnt-1]) * 100
        elif signal_state == 2:
            reward = (close[self.cnt] / close[self.cnt] - 1) * 100
        else:
            reward = 0

        # 변경된 signal_state와 생성된 reward self.inputs에 저장
        self.inputs[self.cnt, self.conf.input_size] = signal_state
        self.inputs[self.cnt, self.conf.input_size+1] = reward

        # input state와 reward 반환
        self.state = tf.convert_to_tensor(self.inputs[self.cnt])
        reward = tf.convert_to_tensor(reward)

        return self.state, reward

    def reset(self, conf) -> tf.Tensor:

        # pred_inpute의 input data와 state_rate_input의 봉간 수익률과 보유 상태를 결합한 input으로 저장
        state_rate, results = state_rate_input(conf)
        self.inputs = np.concatenate([pred_input(conf), state_rate], axis=1)

        self.results = results

        self.max_cnt = len(self.inputs)

        self.cnt = 0

        return tf.convert_to_tensor(self.inputs[0])

# initiate env to apply a step of action
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward = env.step(action)
  return (np.array(state, np.float32),
          np.array(reward, np.float32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action],
                           [tf.float32, tf.float32])

env = env(conf)

def run_episode(
    initial_state: tf.Tensor,
    actor_critic: tf.keras.Model,
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    state = initial_state

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for t in range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        action_probs_t, value = actor_critic(state)
        action = tf.argmax(action_probs_t, 1)[0]

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        action_probs = action_probs.write(t, action_probs_t[0, action])

        # Apply action to the environment to get next state and reward
        state, reward = tf_env_step(action)

        # Store reward
        rewards = rewards.write(t, reward)

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

#@tf.function
def train_step(
    initial_state: tf.Tensor,
    actor_critic: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:

        # 주어진 구간을 하나의 eposode로 하여 차례로 100봉의 수익률 list를 state, 평균 수익률을 reward로 한다.
        # actor_critic모델의 input은 state, output은 action(reinfo_th), value가 된다.
        action_probs, values, rewards = run_episode(initial_state, actor_critic, max_steps_per_episode)

        # expected_returns 계산
        returns = get_expected_return(rewards, gamma)

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

def train(conf, model_name, rwd_th, max_eps, min_eps):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

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
        data.set_ensemble(conf, sorted(random.sample(conf.model_pools, conf.selected_num)))
        conf.pred_term = random.sample([1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40], 1)[0]
        conf.reinfo_width = random.randrange(5, 71, 5)
        conf.loss_cut = random.randint(2, 10) / 2000

        data.set_path(conf, conf.dir)
        env.conf = conf
        env.df = pd.read_csv(conf.df0_path, encoding='euc-kr')
        init_state = env.reset(conf)

        episode_reward = train_step(init_state, actor_critic, optimizer, gamma, env.max_cnt)

        episodes_reward.append(episode_reward.numpy())

        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
            actor_critic.save(model_name)
            pass  # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

    actor_critic.save(model_name)

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')

def test(conf, model_name):

    actor_critic = tf.keras.models.load_model(model_name)

    data.set_path(conf, conf.dir)
    data.set_ensemble(conf, conf.selected_model_types)

    # 데이터 초기화
    env.conf = conf
    env.df = pd.read_csv(conf.df0_path, encoding='euc-kr')
    state = env.reset(conf)

    actions = []
    for i in range(env.max_cnt):
        state = tf.expand_dims(state, 0)

        action_probs_t, _ = actor_critic(state)
        action = tf.argmax(action_probs_t, 1)[0]
        actions.append(action)
        state, _ = tf_env_step(action)

    # 적용전 예측 출력
    conf.result_path = 'test_results.csv'
    p = ep.predict(conf)
    pred_results = pd.read_csv(conf.result_path, encoding='euc-kr').values[:, :6]
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

    rwd_th = 1
    max_eps = 1000
    min_eps = 50

    conf.input_size = 83
    conf.dir = "H:/알고리즘트레이딩2-1"
    model_name = '알고리즘트레이딩2-1/2024-08-01~2025-01-31' + "_rwd_th" + str(rwd_th)

    trained_model = ["10P", "15C", "5P", 0, 10, 20, 0.003]
    conf.reinfo_th = float(trained_model[3])
    conf.pred_term = int(trained_model[4])
    conf.reinfo_width = int(trained_model[5])
    conf.loss_cut = float(trained_model[6])

    if mode == 'train':

        conf.start_time = '2023/01/01/09:00'
        conf.end_time = '2025/01/31/15:00'
        conf.last_train = '2025-01-31'

        #actor_critic = tf.keras.models.load_model(model_name)

        data.set_ensemble(conf, trained_model[:3])
        train(conf, model_name, rwd_th, max_eps, min_eps)

    else:
        conf.start_time = '2025/02/17/09:00'
        conf.end_time = '2025/02/26/15:00'
        conf.last_train = '2025-01-31'

        data.set_ensemble(conf, trained_model[:3])
        test(conf, model_name)
    exit(0)