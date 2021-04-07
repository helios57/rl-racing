import numpy as np
import tensorflow as tf

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from vae_model import CVAE

physical_devices = tf.config.list_physical_devices('GPU')
try:
  print(tf.__version__)
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

from donkey_gym import DonkeyVAEEnv


def train_sac():
  latent_dim = 256
  vae = CVAE(latent_dim)
  vae.load_weights('./vae_256/checkpoint')
  env1 = DonkeyVAEEnv(vae, latent_dim, "Helios1")
  # manual_override=None if you don't want to "help" the Agend with w,a,s,d
  # env1 = DonkeyVAEEnv(vae, latent_dim, "Helios1", manual_override=ManualOverride())
  env1.client.collecting = False
  sac = SAC(env=env1, policy=MlpPolicy, buffer_size=20000, learning_starts=0, train_freq=20000, batch_size=256, verbose=2, gradient_steps=100, learning_rate=0.0005)
  # uncomment if you want to load a model and retrain it
  sac = sac.load("sac/model_sb3", env=env1)
  # sac = sac.load("sac/model_sb3_lake_36", env=env1)
  # sac = sac.load("sac/model_sb3_lake_36_unscaled", env=env1)
  env1.client.hardReset()
  env1.client.initCar()
  env1.client.reset()
  env1.client.restartScene()
  env1.client.hardReset()
  env1.client.initCar()
  env1.client.reset()
  env1.client.collecting = True
  env1.client.telemetrie = []
  while True:
    observation, index = env1.get_observation()
    action = sac.predict(np.asarray([observation]), deterministic=False)[0][0]
    steering, throttle = action[0], action[1]
    env1.client.send_controls(steering * 0.4, throttle)
    # env1.client.send_controls(steering * 0.7, throttle * 0.8)
    print(str(index) + " steering:" + str(action[0]) + " throttle:" + str(action[1]) + " speed:" + str(env1.client.telemetrie[index].speed))


if __name__ == "__main__":
  train_sac()
