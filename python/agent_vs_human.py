from env import Environment
from agent import *
from action import * 
from torch.utils.tensorboard.writer import SummaryWriter
from observation import MatchStatus
import torch
import os

# HYPERPARAMETERS
########################
observations_per_state = 15 
train_matches = 15
eval_matches = 3
batch_size = 128
train_every_steps = 1
learning_rate = 1e-6
memory_capacity = 3000 
update_target_model_every_steps = 2500
discount = 0.95
additional_forbidden_keys = []
additional_forbidden_actions = [] 
save_state = 4

eps_func = lambda x : max(0.2, 0.9 - x / 150000)
########################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = Environment(
  observations_per_state = observations_per_state
)

agent = DDQNAgent(
  eps_func = eps_func,
  observations_per_state = observations_per_state,
  device = device,
  batch_size = batch_size,
  train_every_steps = train_every_steps,
  learning_rate = learning_rate,
  memory_capacity = memory_capacity,
  update_target_model_every_steps = update_target_model_every_steps,
  discount = discount,
  additional_forbidden_keys = additional_forbidden_keys,
  additional_forbidden_actions = additional_forbidden_actions
)

# agent = RandomAgent(
#   additional_forbidden_keys = additional_forbidden_keys,
#   additional_forbidden_actions = additional_forbidden_actions
# )

run_name = "law(easy)(stage1)_FullActionspace_ddqn_target15000_eps150000min0.2max0.9_lr1e-6_bs32_mem3000_tes1_discount0.99_obs15"
run_path = f"runs/{run_name}"
os.makedirs(run_path, exist_ok=True)
writer = SummaryWriter(run_path)

model_path = f"{run_path}/save.lmao"
if os.path.exists(model_path):
  agent.load(model_path)

def standby():
  while True:
    _, info = env.get_observation()
    if info["match_status"] == MatchStatus.ACTIVE and not info["game_paused"]:
      break

def play():
  agent.eval_mode()
  state, _ = env.get_observation(count = observations_per_state, transform = True)
  done = False
  while not done:
    action = agent.generate_action(state)
    new_state, _, done, _ = env.step(action) 
    state = new_state
  env.clear_controller(player = 1)

while True:
  standby()
  print("Match has started, activating agent.")
  play()
