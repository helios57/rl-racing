import uuid

host = "127.0.0.1"
port = 9091
enable_manual_input_during_training = False
manual_input_use_joystick = False
joystick_steering_axis = 0
joystick_throttle_axis = 1
joystick_quit_button = 0
telemetry_data = 'D:/sac_telemetry/'
vae_data = 'vae_torch/vae.pkl'
car_name = 'Helios1'
body_style = 'car01'
body_r = 0
body_g = 255
body_b = 0
backup_telemetry_during_training = False
rl_model = 'sac/model_sb3_torch'
raceline_csv = 'thunderhill.csv'
# raceline_csv = 'thunderhill_optimal_line_stripped.csv'

racer_info = {'msg_type': 'racer_info',
              'racer_name': car_name,
              'car_name': car_name,
              'bio': "IT-Engineer and RC-Enthusiast trying to learn RL with an example like self-driving cars ;-)",
              'country': "Switzerland",
              "guid": str(uuid.uuid4()),
              }
