from env import Environment
from agent import *
from action import * 
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import os
import time

# HYPERPARAMETERS
########################
observations_per_state = 15 
train_matches = 15
eval_matches = 3
batch_size = 32
train_every_steps = 1
learning_rate = 1e-6
memory_capacity = 3000 
update_target_model_every_steps = 15000
discount = 0.99
additional_forbidden_keys = []
additional_forbidden_actions = [] 
save_state = 1

eps_func = lambda x : max(0.2, 0.9 - x / 150000)
########################

# memcap 2500, bs 64

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

run_name = "law(easy)(stage1)_FullActionspace_ddqn_target15000_eps150000min0.2max0.9_lr1e-6_bs32_mem3000_tes1_discount0.99_obs15"
run_path = f"runs/{run_name}"
os.makedirs(run_path, exist_ok=True)
writer = SummaryWriter(run_path)

def train():
  agent.train_mode()
  for _ in range(train_matches):
    state, info = env.reset(save_state)
    done = False
    while not done:
      action = agent.generate_action(state)
      new_state, reward, done, info = env.step(action)
      start = time.time()
      loss = agent.train(state, action, reward, new_state, done, info)
      print(f"Time for train: {(time.time() - start) * 1000} ms")
      if loss >= 0:
        writer.add_scalar("loss", loss, agent.step)
      state = new_state

def evaluate(reset_state = save_state, tag = ""):
  agent.eval_mode()
  wins = 0
  rewards = 0
  dmg_dealt = 0
  dmg_taken = 0
  steps = 0

  with torch.no_grad():
    for _ in range(eval_matches):
      state, info = env.reset(reset_state)
      done = False
      while not done:
        action = agent.generate_action(state)
        new_state, reward, done, info = env.step(action)
        state = new_state
        if "player1_hp_delta" in info:
          dmg_dealt += -info["player2_hp_delta"]
          dmg_taken += -info["player1_hp_delta"]
        rewards += reward
        steps += 1
      wins += info["winner"] == "player1"

  writer.add_scalar("winrate" + tag, wins / eval_matches * 100, agent.evaluations_done)
  writer.add_scalar("avg_dmg_dealt" + tag, dmg_dealt / eval_matches, agent.evaluations_done)
  writer.add_scalar("avg_dmg_taken" + tag, dmg_taken / eval_matches, agent.evaluations_done)
  writer.add_scalar("avg_reward" + tag, rewards / eval_matches, agent.evaluations_done)
  writer.add_scalar("avg_steps" + tag, steps / eval_matches, agent.evaluations_done)
  writer.flush()

model_path = f"{run_path}/save.lmao"
if os.path.exists(model_path):
  agent.load(model_path)

while True:
  agent.save(model_path)
  print(f"Training iteration {agent.evaluations_done} starting.")
  train()
  print(f"Training finished, evaluating agent.")
  evaluate(reset_state = 1, tag = " vs law(easy)")
  evaluate(reset_state = 4, tag = " vs law(hard)")
  agent.evaluations_done += 1
