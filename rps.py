import os.path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import time
import logging
from datetime import datetime, timedelta

import numpy as np
from collections import deque
import random

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActStateModel:
    def __init__(self, state_dim, act_dim, writer=None,
                 eps=args.eps, eps_decay=args.eps_decay, eps_min=args.eps_min):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.writer = writer

        self.model = self._create_model()
        self.model.summary()
        self.optm = Adam(learning_rate=args.lr)
        self.loss = tf.keras.losses.MeanSquaredError()

    def _create_model(self):
        model = Sequential([
            Input((self.state_dim,)),
            Dense(64, activation="relu"),
            Dense(16, activation="relu"),
            Dense(self.act_dim)  # , activation="softmax")
        ])
        # model.compile(loss="mse", optimizer=Adam(args.lr))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_act(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.eps_min)
        q_val = self.predict(state)[0]
        output_act = self.eps_greedy(q_val)
        return output_act

    def eps_greedy(self, q_val):
        if np.random.random() < self.eps:
            print("Explore")
            return random.randint(0, self.act_dim - 1)
        else:
            print("Exploit")
            return np.argmax(q_val)

    def train(self, states, targets, replay=0):
        # self.model.fit(states, targets, epochs=1, verbose=1, callbacks=self.tb_callback)
        epochs = 1
        ds = zip(states, targets)
        for i in range(epochs):
            step_start = time.time()
            with tf.GradientTape() as tape:
                logits = self.model(states, training=True)

                loss = self.loss(targets, logits)

            grads = tape.gradient(loss, self.model.trainable_weights)

            self.optm.apply_gradients(zip(grads, self.model.trainable_weights))

            end_time = time.time() - step_start
            if i == epochs - 1:
                print("Training loss (for one batch) at replay {} epochs {}/{} "
                      ": {loss:.4f} - {time:.3f}s".format(replay, i + 1, epochs, time=end_time, loss=float(loss)))

        return loss


class Agent:
    def __init__(self, env, writer=None, tb=None):
        self.env = env
        self.act_dim = self.env.action_space
        self.writer = writer
        self.tb = tb

        self.state_dim = self.env.state_space
        self.model = ActStateModel(self.state_dim, self.act_dim, writer=self.writer)
        self.target_model = ActStateModel(self.state_dim, self.act_dim, writer=self.writer)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        w = self.model.model.get_weights()
        self.target_model.model.set_weights(w)

    def replay(self):
        for r in range(20):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_vals = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_vals * args.gamma
            loss = self.model.train(states, targets, r)
        return loss

    def sarsa_learn(self, state, act, reward, next_state, done, next_act):
        state = np.float32(state).reshape(1, 3)
        next_state = np.float32(next_state).reshape(1, 3)
        for r in range(20):
            targets = self.model.predict(state)
            next_q_vals = self.model.predict(next_state)
            targets[0, act] = reward if done else (reward + args.gamma * next_q_vals)
            self.model.train(state, targets, r)

    def train_dqn(self, max_ep=1000, render_ep=100):
        for ep in range(max_ep):
            done = False
            total_reward = 0
            state = self.env.reset()

            while not done:
                act = self.model.get_act(state)
                next_state, reward, done = self.env.step(act)
                self.buffer.put(state, act, reward * 0.01, next_state, done)
                total_reward += reward

            if self.buffer.size() >= args.batch_size:
                loss = self.replay()
                with self.writer.as_default():
                    tf.summary.scalar("Loss per ep", loss, step=ep)
            self.target_update()

            print('EP{} EpisodeReward={} Epsilon={}'.format(ep, total_reward, self.model.eps))

            with self.writer.as_default():
                tf.summary.scalar("Episode_Reward", total_reward, step=ep)
                tf.summary.scalar("Epsilon", self.model.eps, step=ep)

            if (ep + 1) % render_ep == 0:
                state = self.env.reset()
                while not done:
                    self.env.render()
                    act = self.model.get_act(state)
                    next_state, reward, done = self.env.step(act)


class RPSEnv(gym.Env):
    tags = ["ROCK", "PAPER", "SCISSORS"]
    """
    win = [1,0,0]
    lose = [0,1,0]
    draw = [0,0,1]

    ROCK = [1,0,0]
    PAPER = [0,1,0]
    SCISSORS = [0,0,1]
    """

    def __init__(self):
        self.state = None
        self.result = [0, 0, 1]
        self.state_space = 3
        self.action_space = 3
        self.win_count = 0
        self.win_target_count = 3

        self.loses_to = {
            "0": 1,  # rock 2 paper
            "1": 2,  # paper 2 scissors
            "2": 0  # scissors 2 rock
        }

        self.result_str = {
            "0": "WIN",
            "1": "LOSE",
            "2": "DRAW"
        }

    def state2str(self, result):
        return self.result_str[str(np.argmax(result))]

    def get_result(self, player_move, bot_move):
        player_move = np.argmax(player_move)
        bot_move = np.argmax(bot_move)
        result = np.zeros(self.state_space)

        if bot_move == player_move:
            result[-1] = 1  # DRAW
        elif self.loses_to[str(bot_move)] == player_move:
            result[1] = 1  # LOSE
        else:
            result[0] = 1  # WIN

        return result

    def get_reward(self, state, act):
        self.result = self.get_result(state, act)
        res = np.argmax(self.result)

        if res == 0:
            reward = 3
            self.win_count += 1
        elif res == 1:
            reward = -10
        else:
            reward = 1

        if self.win_count == self.win_target_count:
            reward += 5

        return reward

    def step(self, act):
        reward = self.get_reward(self.state, act)
        # done = True if (self.win_count == self.win_target_count) else False
        done = True
        # if self.win_count == self.win_target_count:
        #     done = True

        return self.result, reward, done  # , self.win_count, self.win_target_count

    def reset(self):
        tmp = np.random.randint(0, self.action_space)
        self.state = np.zeros(self.action_space)
        self.state[tmp] = 1
        self.win_count = 0
        return self.state

    def state_random_sampling(self):
        tmp = np.random.randint(0, self.action_space)
        state = np.zeros(self.action_space)
        state[tmp] = 1
        return state

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def main():
    start_datetime = datetime.now()
    start_time = time.monotonic()

    log_path = "logs2/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(log_path):
        print("Creating [{}]".format(log_path))
        os.makedirs(log_path)

    # tb = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    summary_writer = tf.summary.create_file_writer(log_path)

    env = RPSEnv()
    agent = Agent(env, summary_writer)
    agent.train_sarsa(1000, 100)

    end_time = time.monotonic()
    end_datetime = datetime.now()

    duration = timedelta(seconds=end_time - start_time)
    print("Experiment start at: {}".format(start_datetime))
    print("Experiment End Time: {}".format(end_datetime))
    print("Time taken: {}".format(duration))


if __name__ == "__main__":
    main()
