import cv2
import numpy as np


def picture_filter(img):
  original = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  blur = cv2.GaussianBlur(original, (11, 11), 0)
  imghsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype("float32")
  (h, s, v) = cv2.split(imghsv)
  s = s * 2
  s = np.clip(s, 0, 255)
  v = v * 2
  v = np.clip(v, 0, 255)
  imghsv = cv2.merge([h, s, v])
  return cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
