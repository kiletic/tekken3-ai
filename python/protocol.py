from observation import * 
from agent import *
from enum import IntEnum
from PIL import Image
import struct

OPCODE_SIZE = 1 # in bytes

class ParseType(IntEnum):
  OBSERVATION_START = 0
  SCREENSHOT = 1
  MATCH_STATUS = 2
  GAME_PAUSED = 3
  PLAYER1_HP = 4
  PLAYER2_HP = 5
  OBSERVATION_END = 255 

class SendType(IntEnum):
  ACTION = 1
  RESET = 2
  OBSERVATIONS_REQUEST = 3
  CLEAR_CONTROLLER = 4

class NetworkParser:
  def __init__(self, connection):
    self.connection = connection

  def parse_observation(self):
    assert self.readU8() == ParseType.OBSERVATION_START
    observation = Observation()
    while (opcode := self.readU8()) != ParseType.OBSERVATION_END:
      match opcode:
        case ParseType.SCREENSHOT:
          observation.frame = self.parse_screenshot()
        case ParseType.MATCH_STATUS:
          observation.match_status = self.parse_match_status()
        case ParseType.GAME_PAUSED:
          observation.game_paused = self.parse_game_paused()
        case ParseType.PLAYER1_HP:
          observation.player1_hp = self.parse_player1_hp()
        case ParseType.PLAYER2_HP:
          observation.player2_hp = self.parse_player2_hp()
        case _:
          raise ValueError(f"Received invalid opcode: {opcode}")
    return observation

  def parse_match_status(self):
    match_status = self.readU8() 
    # 0 -> not in match
    # 1 -> in match
    # 2 -> match finished
    assert match_status >= 0 and match_status <= 2, f"Parsed invalid match_status, its equal to {match_status}"
    return match_status

  def parse_game_paused(self):
    game_paused = self.readU8() 
    # 0 -> not paused
    # 1 -> paused
    assert game_paused == 0 or game_paused == 1, f"Parsed invalid game_paused, its equal to {game_paused}"
    return game_paused 
  
  def parse_player1_hp(self):
    player1_hp = self.readU8() 
    # player1 max hp is 200
    # assert player1_hp >= 0 and player1_hp <= 200, f"Parsed invalid player1_hp, its equal to {player1_hp}"
    return player1_hp 

  def parse_player2_hp(self):
    player2_hp = self.readU8() 
    # player2 max hp is 200 
    # assert player2_hp >= 0 and player2_hp <= 200, f"Parsed invalid player2_hp, its equal to {player2_hp}"
    return player2_hp 

  def parse_screenshot(self):
    height = self.readI32() 
    assert height > 0, f"Parsed invalid height from screenshot, its equal to {height}"

    width = self.readI32()
    assert width > 0, f"Parsed invalid width from screenshot, its equal to {width}"

    bpp = self.readU8() 
    assert bpp == 16 or bpp == 24, f"Parsed invalid bpp from screenshot, its equal to {bpp}"

    screenshot_len = self.readI32() 
    screenshot = self.read_raw(screenshot_len)

    mode = "RGB" if bpp == 24 else "RGB;15"
    return Image.frombytes("RGB", (width, height), bytes(screenshot), 'raw', mode)

  def send_action(self, action, player):
    # first byte opcode
    # second byte 1 or 2 indicating player 1 or 2
    # third and fourth byte action
    assert player == 1 or player == 2, f"Trying to send invalid player, its equal to {player}"
    packet = struct.pack('!BBBB', SendType.ACTION, player, *action)
    self.connection.sendall(packet)

  def send_reset(self, state):
    packet = struct.pack('!BB', SendType.RESET, state)
    self.connection.sendall(packet)

  def send_observations_request(self, num_of_observations):
    packet = struct.pack('!BB', SendType.OBSERVATIONS_REQUEST, num_of_observations)
    self.connection.sendall(packet)
    
  def send_clear_controller(self, player):
    packet = struct.pack('!BB', SendType.CLEAR_CONTROLLER, player)
    self.connection.sendall(packet)

  def readU8(self):
    return ord(self.connection.recv(OPCODE_SIZE))

  def readI32(self):
    return int.from_bytes(self.connection.recv(4), byteorder="little")

  def read_raw(self, length):
    data = bytearray()
    while len(data) < length:
      chunk = self.connection.recv(length - len(data))
      data.extend(chunk)
    return data
