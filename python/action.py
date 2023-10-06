import random
from enum import IntEnum

# as per pcsx-redux
class Keyboard(IntEnum):
  SELECT = 0
  START = 3
  UP = 4
  RIGHT = 5
  DOWN = 6
  LEFT = 7
  L2 = 8
  R2 = 9
  L1 = 10
  R1 = 11
  TRIANGLE = 12
  CIRCLE = 13
  CROSS = 14
  SQUARE = 15
  DO_NOTHING = 16 # custom

class ActionSpace:
  def __init__(self, additional_forbidden_keys = [], additional_forbidden_actions = []):
    self.action_to_num = {}
    self.num_to_action = {}
    self.forbidden_keys = [
      Keyboard.L1, Keyboard.L2, Keyboard.R1, Keyboard.R2, Keyboard.SELECT, Keyboard.START
    ] + additional_forbidden_keys
    self.forbidden_actions = [
      (Keyboard.UP, Keyboard.DOWN),
      (Keyboard.RIGHT, Keyboard.LEFT),
      # duplicates redundant since DO_NOTHING exists 
      (Keyboard.UP, Keyboard.UP),
      (Keyboard.DOWN, Keyboard.DOWN),
      (Keyboard.LEFT, Keyboard.LEFT),
      (Keyboard.RIGHT, Keyboard.RIGHT),
      (Keyboard.TRIANGLE, Keyboard.TRIANGLE),
      (Keyboard.CROSS, Keyboard.CROSS),
      (Keyboard.SQUARE, Keyboard.SQUARE),
      (Keyboard.CIRCLE, Keyboard.CIRCLE)
    ] + additional_forbidden_actions
    self.additional_forbidden_keys = additional_forbidden_keys
    self.additional_forbidden_actions = additional_forbidden_actions
    
    self.generate_mappings()
    self.NO_ACTION = self.action_to_num[(Keyboard.DO_NOTHING, Keyboard.DO_NOTHING)]

  def __len__(self):
    return len(self.action_to_num)

  def generate_random_action(self):
    def generate_valid_key():
      key = random.randint(Keyboard.UP, Keyboard.DO_NOTHING)
      while key in self.forbidden_keys:
        key = random.randint(Keyboard.UP, Keyboard.DO_NOTHING)
      return key

    first_key = generate_valid_key() 
    second_key = generate_valid_key()

    if first_key > second_key:
      first_key, second_key = second_key, first_key

    while (first_key, second_key) in self.forbidden_actions:
      second_key = generate_valid_key()
      if first_key > second_key:
        first_key, second_key = second_key, first_key

    print(f"Random action: {(first_key, second_key)}")
    return (first_key, second_key)

  def generate_mappings(self):
    for first_key in Keyboard:
      if first_key in self.forbidden_keys:
        continue
      for second_key in Keyboard:
        if second_key.value < first_key.value or \
           second_key in self.forbidden_keys or \
           (first_key, second_key) in self.forbidden_actions:
          continue
        self.action_to_num[(first_key.value, second_key.value)] = len(self.action_to_num)
    self.num_to_action = {v: k for k, v in self.action_to_num.items()}
    return self.action_to_num, self.num_to_action
