import torch
from stable_baselines3.common.callbacks import BaseCallback

from config import vae_data, rl_model
from vae import TorchVAE

print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

from rl_gym import RLEnv


def train_sac():
  vae = TorchVAE()
  vae.cuda()
  torch.no_grad()
  vae.weight_init(mean=0, std=0.02)
  vae.load_state_dict(torch.load(vae_data))

  env1 = RLEnv(vae)
  env1.client.collecting = False
  policy_kwargs = {}
  policy_kwargs["net_arch"] = [512, 512, 512]
  sac = SAC(env=env1, policy=MlpPolicy,
            policy_kwargs=policy_kwargs,
            buffer_size=20000,
            learning_starts=0,
            train_freq=(1, "episode"),
            batch_size=256,
            verbose=2,
            gradient_steps=-1,
            learning_rate=0.001)
  # uncomment if you want to load a model and retrain it
  sac = sac.load(rl_model, env=env1)
  sac.learning_rate = 0.001
  sac._setup_lr_schedule()
  env1.client.hardReset()
  env1.client.initCar()
  env1.client.reset()
  # env1.client.restartScene()
  env1.client.hardReset()
  env1.client.initCar()
  env1.client.reset()
  env1.client.collecting = True
  env1.client.telemetrie = []
  callback = CustomCallback(env1)
  while True:
    sac.learn(20000, callback=callback)
    sac.save(rl_model)


class CustomCallback(BaseCallback):
  def __init__(self, env, verbose=0):
    super(CustomCallback, self).__init__(verbose)
    self.env = env

  def _on_rollout_start(self) -> None:
    self.env.client.hardReset()
    self.env.client.initCar()
    self.env.client.reset()
    self.env.client.collecting = True
    self.env.client.telemetrie = []
    pass

  def _on_step(self) -> bool:
    return True


if __name__ == "__main__":
  train_sac()
