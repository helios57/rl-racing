import pickle
import random
import zlib
from os import walk

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

from config import telemetry_data, vae_data
from vae import TorchVAE, process_batch, loss_function

BATCH_SIZE = 256


def load_data(max=50000):
  files = []
  for (dirpath, dirnames, filenames) in walk(telemetry_data):
    for filename in filenames:
      files.append(dirpath + filename)
  files.sort(reverse=True)
  images = []
  for file in files:
    with open(file, 'rb') as episode_file:
      telemetries = pickle.loads(zlib.decompress(episode_file.read()))
      for tele in telemetries:
        images.append(tele.img)
        # images.append(numpy.asarray(tele.img * 255, dtype='uint8'))
        if len(images) >= max:
          break
  return images


def train():
  epochs = 300
  batch_size = 128
  vae = TorchVAE()
  vae.cuda()
  vae.train()
  vae.weight_init(mean=0, std=0.02)
  try:
    vae.load_state_dict(torch.load(vae_data))
    print("loaded state")
  except:
    pass

  lr = 0.0001
  vae_optimizer = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
  for epoch in range(epochs):
    vae.train()
    data_train = load_data()
    print("Train set size:", len(data_train))
    random.shuffle(data_train)
    batches = []
    for i in range(0, len(data_train) - batch_size, batch_size):
      batches.append(np.asarray(data_train[i:i + batch_size]))
    i = 0
    for x in batches:
      batch = process_batch(x)
      vae.train()
      vae.zero_grad()
      rec, mu, logvar = vae(batch)

      loss_re, loss_kl = loss_function(rec, batch, mu, logvar)
      (loss_re + loss_kl).backward()
      vae_optimizer.step()

      i += 1
      if i % 32 == 0:
        with torch.no_grad():
          vae.eval()
          x_rec, _, _ = vae(batch[:8])
          resultsample = torch.cat([batch[:8], x_rec]) * 0.5 + 0.5
          resultsample = resultsample.cpu()
          ndarr = make_grid(resultsample.view(-1, 3, 128, 128)).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
          cv2.imshow("rec", cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR))
          cv2.waitKey(1)
      del batch
      torch.cuda.empty_cache()

    del batches
    del data_train
    torch.save(vae.state_dict(), vae_data)
  print("Training finish!... save training results")


if __name__ == "__main__":
  train()
