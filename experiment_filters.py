import pickle
import random
import zlib
from os import walk

import cv2
import numpy as np

from config import telemetry_data


def load_data(max=50000):
  files = []
  for (dirpath, dirnames, filenames) in walk(telemetry_data):
    for filename in filenames:
      files.append(dirpath + filename)
  # files.sort(reverse=True)
  random.shuffle(files)
  images = []
  for file in files:
    with open(file, 'rb') as episode_file:
      telemetries = pickle.loads(zlib.decompress(episode_file.read()))
      for tele in telemetries:
        images.append(tele.img)
        if len(images) >= max:
          return images
      del telemetries
  return images


def train():
  epochs = 300
  for epoch in range(epochs):
    data_train = load_data(5000)
    print("Train set size:", len(data_train))
    for x in data_train:
      original = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
      # cv2.imshow("original", original)
      blur = cv2.GaussianBlur(original, (11, 11), 0)
      # cv2.imshow("blur", blur)
      imghsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype("float32")
      (h, s, v) = cv2.split(imghsv)
      s = s * 2
      s = np.clip(s, 0, 255)
      v = v * 2
      v = np.clip(v, 0, 255)
      imghsv = cv2.merge([h, s, v])
      imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
      cv2.imshow("imgrgb", imgrgb)
      cv2.waitKey(0)
  del data_train
  print("Training finish!... save training results")


if __name__ == "__main__":
  train()
