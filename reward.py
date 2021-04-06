from math import fabs

import numpy as np
import pandas as pd

from config import raceline_csv

race_line = None
a = None
b = None


def load_race_line():
  global race_line
  global a
  global b
  dataset = pd.read_csv('optimal_lines/' + raceline_csv, sep=',')
  race_line = np.asarray(dataset[['x', 'z']].values)
  a = race_line[:-1]
  b = race_line[1:]


def reward_function(telemetrie, index, steering, throttle):
  if index < 2:
    return 0
  current = telemetrie[index]
  before = telemetrie[index - 2]
  speed_vec = np.asarray([current.pos_x - before.pos_x, current.pos_z - before.pos_z])
  race_line, vec, race_line_segment_index = get_distance_to_race_line(current.pos_x, current.pos_z)
  race_line_reward = (1 - (fabs(race_line) / 20.0))

  direction_reward = 1
  if np.linalg.norm(vec) > 0 and np.linalg.norm(speed_vec) > 0:
    velocity_diff = vec / np.linalg.norm(vec) - speed_vec / np.linalg.norm(speed_vec)
    direction_reward = 1 - np.sum(np.absolute(velocity_diff)) / 2

  jerk_penalty = 0
  if steering is not None:
    future_index = (race_line_segment_index[0] + int(current.speed)) % len(b)
    future_heading_vec = b[future_index] - a[future_index]
    if np.linalg.norm(future_heading_vec) > 0 and np.linalg.norm(speed_vec) > 0:
      try:
        future_heading_vec_normalized = future_heading_vec / np.linalg.norm(future_heading_vec)
        speed_vec_normalized = speed_vec / np.linalg.norm(speed_vec)
        future_velocity_diff_normalized = future_heading_vec_normalized - speed_vec_normalized
        jerk_penalty = - fabs(np.cross(speed_vec_normalized, future_velocity_diff_normalized) - steering)
      except:
        pass

  # reward beeing near to line and heading in line direction
  reward = (race_line_reward * direction_reward * current.speed) / 10.0
  if race_line_reward < 0 and direction_reward < 0:
    reward = -reward

  # punish wrong steering directly
  reward = reward + jerk_penalty * 0.005

  print(index, reward, race_line_reward, direction_reward, jerk_penalty)
  return reward


def get_distance_to_race_line(x, z):
  if race_line is None:
    load_race_line()
  dists = lineseg_dists(np.asarray([x, z]), a, b)
  amin = np.amin(dists)
  index = np.where(dists == amin)[0]
  vec = b[index] - a[index]
  return amin, vec, index


def lineseg_dists(p, a, b):
  """Cartesian distance from point to line segment

  Edited to support arguments as series, from:
  https://stackoverflow.com/a/54442561/11208892

  Args:
      - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
      - a: np.array of shape (x, 2)
      - b: np.array of shape (x, 2)
  """
  # normalized tangent vectors
  d_ba = b - a
  d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                       .reshape(-1, 1)))

  # signed parallel distance components
  # rowwise dot products of 2D vectors
  s = np.multiply(a - p, d).sum(axis=1)
  t = np.multiply(p - b, d).sum(axis=1)

  # clamped parallel distance
  h = np.maximum.reduce([s, t, np.zeros(len(s))])

  # perpendicular distance component
  # rowwise cross products of 2D vectors
  d_pa = p - a
  c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

  return np.hypot(h, c)


def calc_reward(telemetrie, index, steering, throttle):
  if telemetrie[index].totalNodes <= 0.01:
    return 0
  return reward_function(telemetrie, index, steering, throttle)
