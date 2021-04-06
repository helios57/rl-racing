1. Install requirements

* `python -m pip install -r requirements.txt`

2. check / update config.py & create referenced directories

* mkdir sac_telemetry
* mkdir vae_torch

3. Start the Sim (https://github.com/tawnkramer/sdsandbox) and open the track, you may click on NN Control over Network (optional but removes the dummy car)

4. gather telemetry data by manual drive the course. To gather data for the vae you do not need to drive clean, make sure you get all possible angles/pictures possible, I would recommend about 3 laps
   or more

* `gather_manual_data.py` press "q" or the definied joystick quit button to save the recorded data

5. train the vae (variational autoencoder) with the gathered data

* `train_vae.py`

6. add a optimal raceline (or a manual one) and load it in the rewards, if you take the log-output from the `gather_manual_data.py` you can use the line you drove as a raceline, just take the points
   between 2 laps

* `reward.py`

7. train the RL-Agent

* `train_rl_agent.py`

8. adjust the reward

* `reward.py`

TODO:

* I had extended telemetry turned off, so perhaps a quick telemetry frame check and give error message "Please turn on Extended Telemetry"
