import pickle
import time
import zlib
from math import fabs

from config import telemetry_data, host, port
from donkey_client import DonkeyClient
from manual_input import ManualInput


def test_clients():
  manual_input = ManualInput()
  client = DonkeyClient(address=(host, port))
  client.collecting = False
  client.hardReset()
  client.initCar()
  client.reset()
  client.collecting = True
  client.telemetrie = []
  while len(client.telemetrie) < 10:
    time.sleep(0.1)
  last_printed_index = 0
  while True:
    if manual_input.q == 1:
      client.collecting = False
      break
    current_telemetrie = client.telemetrie[len(client.telemetrie) - 1]
    manual_input.loop(current_telemetrie.img)
    if (len(client.telemetrie) % 20 == 0):
      index = len(client.telemetrie)
      if last_printed_index != index:
        print(current_telemetrie.pos_x, ",", current_telemetrie.pos_y, ",", current_telemetrie.pos_z)
        last_printed_index = index
    if manual_input.e == 1:
      th1 = 0.8
      if fabs(current_telemetrie.cte) > 0.5:
        th1 = 0.2
      elif fabs(current_telemetrie.cte) > 1:
        th1 = 0.5
      elif fabs(current_telemetrie.cte) > 2:
        th1 = 0.3
      client.send_controls(-current_telemetrie.cte / 2.0, th1)
    else:
      client.send_controls(manual_input.st, manual_input.th)

  with open(telemetry_data + str(time.time_ns()) + '.pk', 'wb') as episode_file:
    episode_file.write(zlib.compress(pickle.dumps(client.telemetrie)))

  time.sleep(1.0)
  client.stop()


if __name__ == "__main__":
  test_clients()
