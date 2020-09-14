import time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from config import host, port, enable_manual_input_during_training
from donkey_client import DonkeyClient
from manual_input import ManualInput
from reward import calc_reward


class RLEnv(gym.Env):
  def __init__(self, vae):
    self.stackSize = 2
    self.latent_dim = vae.zsize
    self.vae = vae
    self.client = DonkeyClient(address=(host, port))
    self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

    # stacked self.stackSize latest observations images encoded = latent_dim(picture) + (steering + throttle + speed) + (accel x,y,z) + gyro (x,y,z,w)
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.stackSize, self.latent_dim + 3 + 3 + 4 + 3), dtype=np.float32)
    self.seed()
    self.index = 0
    self.manual_input = None
    if enable_manual_input_during_training:
      self.manual_input = ManualInput()
    self.reward_accumulator = 1

  def step(self, action):
    """
    :param action: (np.ndarray)
    :return: (np.ndarray, float, bool, dict)
    """
    if self.manual_input is not None:
      self.manual_input.loop(self.client.telemetrie[len(self.client.telemetrie) - 1].img)
    throttle = action[1]  # * 0.35 + 0.65
    if throttle > 0.5:
      throttle = 1
    steering = action[0] ** 3  # make steering exponential
    if self.manual_input is not None:
      printstr = str(throttle) + " " + str(steering) + " "
      if self.manual_input.th == 1:
        throttle = 1
        steering = 0
      if self.manual_input.th == -1:
        throttle = -1
      if self.manual_input.st == 1:
        steering = 1
      if self.manual_input.st == -1:
        steering = -1
      if self.manual_input.e == 1:
        print(printstr, throttle, steering)
      steering = steering

    self.client.send_controls(steering, throttle)
    return self.observe(steering, throttle)

  def reset(self, soft=True):
    if soft:
      self.client.softReset()
    else:
      self.client.reset()
    observation, reward, done, info = self.observe()
    return observation

  def get_observation(self, data=None, index=None):
    if index is None:
      while len(self.client.telemetrie) < self.stackSize or self.index == len(self.client.telemetrie) - 1:
        time.sleep(0.01)
      index = len(self.client.telemetrie) - 1
      if self.index + 3 < index:
        print("lost frames " + str(index - self.index - 1))
      self.index = index
    if data is None:
      data = self.client.telemetrie
    if data[index].obs is not None:
      return data[index].obs, index
    stack = data[index - self.stackSize + 1:index + 1]
    imgs = np.asarray([o.img for o in stack])
    tele = np.asarray(
      [[o.steering_angle,
        o.throttle,
        o.speed / 40.0,
        o.accel_x / 20.0,
        o.accel_y / 20.0,
        o.accel_z / 20.0,
        o.gyro_x,
        o.gyro_y,
        o.gyro_z,
        o.gyro_w,
        o.euler[0],
        o.euler[1],
        o.euler[2]] for o in stack])
    z = self.vae.encode_to_z_numpy(imgs)
    imgEncoded = z  # self.vae.encode(imgs).numpy()
    obs = np.concatenate([imgEncoded, tele], axis=1)
    data[index].obs = obs
    del imgEncoded
    del tele
    del imgs
    del stack
    del data

    return obs, index

  def observe(self, steering=0, throttle=0):
    observation, index = self.get_observation()
    reward = calc_reward(self.client.telemetrie, index, steering, throttle)
    self.reward_accumulator = self.reward_accumulator * 0.9 + reward * 0.1
    done = index > 100 and (self.reward_accumulator < 0 or self.client.telemetrie[index].speed < 2)
    if done:
      reward = -1
    info = {"telemetry": self.client.telemetrie[index],
            "action": np.asarray([self.client.telemetrie[index].steering_angle, self.client.telemetrie[index].throttle])}
    return observation, reward, done, info

  def close(self):
    self.client.stop()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
