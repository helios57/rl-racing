import base64
import gc
import json
import pickle
import time
import zlib
from io import BytesIO

import numpy as np
from PIL import Image
from gym_donkeycar.core.sim_client import SDClient

from config import telemetry_data, backup_telemetry_during_training, racer_info, \
  body_style, body_r, body_g, body_b, car_name


def quaternion_to_euler(w, x, y, z):
  """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
  sinr_cosp = 2 * (w * x + y * z)
  cosr_cosp = 1 - 2 * (x ** 2 + y ** 2)
  roll = np.arctan2(sinr_cosp, cosr_cosp)

  sinp = 2 * (w * y - z * x)
  pitch = np.where(np.abs(sinp) >= 1,
                   np.sign(sinp) * np.pi / 2,
                   np.arcsin(sinp))

  siny_cosp = 2 * (w * z + x * y)
  cosy_cosp = 1 - 2 * (y ** 2 + z ** 2)
  yaw = np.arctan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw


class DonkeyClient(SDClient):
  class Telemetry:
    def __init__(self):
      self.img = None
      self.imgEncoded = None
      self.steering_angle = None
      self.throttle = None
      self.speed = None
      self.velocity = None
      self.accel_x = None
      self.accel_y = None
      self.accel_z = None
      self.pos_x = None
      self.pos_y = None
      self.pos_z = None
      self.gyro_x = None
      self.gyro_y = None
      self.gyro_z = None
      self.gyro_w = None
      self.euler = None
      self.hit = None
      self.cte = None
      self.activeNode = None
      self.totalNodes = None
      self.obs = None
      self.delay = None

  def __init__(self, address, poll_socket_sleep_time=0.01):
    super().__init__(*address, poll_socket_sleep_time=poll_socket_sleep_time)
    self.collecting = False
    self.car_loaded = False
    self.race_mode = False
    self.telemetrie = []
    self.debug = False
    self.collision_with_starting_line = 0
    self.collision_with_starting_line_last = 0

  def on_msg_recv(self, json_packet):
    if json_packet is None or len(json_packet) == 0:
      return
    if json_packet['msg_type'] == "need_car_config":
      self.initCar()
      return
    if json_packet['msg_type'] == "car_loaded":
      self.car_loaded = True
      return
    if json_packet['msg_type'] == "telemetry":
      if self.collecting:
        image = Image.open(BytesIO(base64.b64decode(json_packet["image"])))
        del json_packet["image"]
        telemetry = self.Telemetry()
        telemetry.img = np.asarray(image).astype('uint8')
        telemetry.steering_angle = json_packet["steering_angle"]
        telemetry.throttle = json_packet["throttle"]
        telemetry.speed = json_packet["speed"]
        telemetry.accel_x = json_packet["accel_x"]
        telemetry.accel_y = json_packet["accel_y"]
        telemetry.accel_z = json_packet["accel_z"]
        telemetry.pos_x = json_packet["pos_x"]
        telemetry.pos_y = json_packet["pos_y"]
        telemetry.pos_z = json_packet["pos_z"]
        telemetry.gyro_x = json_packet["gyro_x"]
        telemetry.gyro_y = json_packet["gyro_y"]
        telemetry.gyro_z = json_packet["gyro_z"]
        telemetry.gyro_w = json_packet["gyro_w"]
        telemetry.euler = quaternion_to_euler(telemetry.gyro_w, telemetry.gyro_x, telemetry.gyro_y, telemetry.gyro_z)
        if not self.race_mode:
          if self.debug and len(self.telemetrie) % 20 == 0:
            print(str(json_packet["pos_x"]) + ',' + str(json_packet["pos_y"]) + ',' + str(json_packet["pos_z"]))
            print(str(json_packet["cte"]))
          telemetry.velocity = np.asarray([float(json_packet["vel_x"]), float(json_packet["vel_y"]), float(json_packet["vel_z"])])
          telemetry.hit = json_packet["hit"]
          telemetry.cte = json_packet["cte"]
          telemetry.activeNode = json_packet["activeNode"]
          telemetry.totalNodes = json_packet["totalNodes"]
        self.telemetrie.append(telemetry)
        del json_packet
      return
    if json_packet['msg_type'] == "ping":
      return
    if json_packet['msg_type'] == "cross_start":
      return
    if json_packet['msg_type'] == "collision_with_starting_line":
      if self.collision_with_starting_line < 0:
        self.collision_with_starting_line = json_packet['timeStamp']
        return
      self.collision_with_starting_line_last = self.collision_with_starting_line
      self.collision_with_starting_line = json_packet['timeStamp']
      print("lap time ", (self.collision_with_starting_line - self.collision_with_starting_line_last))
      return
    print(json_packet)

  def send_controls(self, steering, throttle):
    self.send(json.dumps({"msg_type": "control", "steering": str(steering), "throttle": str(throttle), "brake": "0.0"}))
    time.sleep(self.poll_socket_sleep_sec)

  def initCar(self):
    time.sleep(0.5)
    self.send(json.dumps(racer_info))
    time.sleep(0.5)
    self.send(
      '{ "msg_type" : "car_config", "body_style" : "' + body_style + '", "body_r" : "' + str(body_r) + '", "body_g" : "' + str(body_g) + '", "body_b" : "' + str(
        body_b) + '", "car_name" : "' + car_name + '", "font_size" : "30" }')
    time.sleep(self.poll_socket_sleep_sec)
    time.sleep(0.5)
    self.send(
      '{ "msg_type" : "cam_config", "fov" : "100", "fish_eye_x" : "0.0", "fish_eye_y" : "0.0", "img_w" : "128", "img_h" : "128", "img_d" : "3", "img_enc" : "PNG", "offset_x" : "0.0", "offset_y" : "2.5", "offset_z" : "0.5", "rot_x" : "20.0" }')
    time.sleep(self.poll_socket_sleep_sec)
    time.sleep(0.5)

  def softReset(self):
    self.backupTelemtetrie()
    self.send(json.dumps({'msg_type': 'reset_car'}))
    time.sleep(self.poll_socket_sleep_sec)
    del self.telemetrie
    self.telemetrie = []
    gc.collect()

  def hardReset(self):
    self.backupTelemtetrie()
    del self.telemetrie
    self.telemetrie = []
    gc.collect()

  def reset(self):
    self.backupTelemtetrie()
    self.send(json.dumps({'msg_type': 'reset_car'}))
    time.sleep(1)
    self.send_controls(0, 0)
    del self.telemetrie
    self.telemetrie = []
    self.collision_with_starting_line = -1
    gc.collect()

  def backupTelemtetrie(self):
    if self.telemetrie is None or len(self.telemetrie) < 100 or not backup_telemetry_during_training:
      return
    with open(telemetry_data + str(time.time_ns()) + '.pk', 'wb') as episode_file:
      episode_file.write(zlib.compress(pickle.dumps(self.telemetrie)))
