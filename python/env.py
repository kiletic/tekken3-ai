from protocol import NetworkParser
from server import Server
from observation import MatchStatus
import numpy as np
import torch
import cv2

HOST = '127.0.0.1'
PORT = 50007

class Environment:
  def __init__(self, observations_per_state = 15):
    self.server = Server(HOST, PORT)
    self.observations_per_state = observations_per_state
    self.last_observation = None

    connection = self.server.connect()
    self.network_parser = NetworkParser(connection)

  def reset(self, state):
    self.network_parser.send_reset(state)
    self.gather_observations()

    state = self.transform_observations(self.observations)
    info = self.extract_info(self.observations, self.last_observation)
    
    self.last_observation = self.observations[-1]
    return state, info 

  def step(self, action, player = 1):
    self.gather_observations()
    state = self.transform_observations(self.observations)
    info = self.extract_info(self.observations, self.last_observation)
      
    # delay of observations_per_state frames to account for human reaction time
    # if self play then action will be a list 
    if type(action) == list:
      player1_action, player2_action = action
      self.network_parser.send_action(player1_action, player = 1)
      self.network_parser.send_action(player2_action, player = 2)
    else:
      self.network_parser.send_action(action, player)

    reward = self.calculate_reward(info, player)
    self.last_observation = self.observations[-1]
    return state, reward, info["done"], info

  def gather_observations(self):
    self.observations = []
    self.network_parser.send_observations_request(self.observations_per_state)
    while len(self.observations) != self.observations_per_state:
      self.observations.append(self.network_parser.parse_observation())

  def get_observation(self, count = 1, transform = False):
    observations = []
    self.network_parser.send_observations_request(count)
    while len(observations) != count:
      observations.append(self.network_parser.parse_observation())

    info = self.extract_info(observations)
    if transform:
      observations = self.transform_observations(observations) 
    if count == 1:
      observations = observations[0]

    return observations, info 

  def extract_info(self, observations, last_observation = None):
    info = {
      "match_status" : observations[-1].match_status,
      "game_paused" : observations[-1].game_paused,
      "player1_hp" : observations[-1].player1_hp,
      "player2_hp" : observations[-1].player2_hp,
      "done" : observations[-1].match_status == MatchStatus.FINISHED #or min(observations[-1].player1_hp, observations[-1].player2_hp) == 0
    }

    if last_observation:
      info["player1_hp_delta"] = observations[-1].player1_hp - last_observation.player1_hp
      info["player2_hp_delta"] = observations[-1].player2_hp - last_observation.player2_hp

    if info["done"]:
      if info["player1_hp"] > info["player2_hp"]:
        info["winner"] = "player1"
      else:
        info["winner"] = "player2"

    return info

  def transform_observations(self, observations):
    assert len(observations) > 0
    frames = [observation.frame.resize((102, 76)) for observation in observations]
    frames = np.array([cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY) for frame in frames])
    state = torch.tensor(frames, dtype=torch.float32) / 255

    return state
  
  def calculate_reward(self, info, player):
    reward_hp = 0
    if "player1_hp_delta" in info:
      dmg_received = -info["player1_hp_delta"]
      dmg_dealt = -info["player2_hp_delta"]
      reward_hp = dmg_dealt - dmg_received

    reward_end = 0
    if info["done"]:
      if info["player1_hp"] > info["player2_hp"]:
        reward_end = 100
      else:
        reward_end = -100

    # if we're calculating reward for player 2
    if player == 2:
      reward_hp *= -1
      reward_end *= -1

    return reward_hp + reward_end

  def clear_controller(self, player):
    self.network_parser.send_clear_controller(player)
