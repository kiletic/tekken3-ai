from env import Environment
from agent import *
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import os
import time

# HYPERPARAMETERS
########################
observations_per_state = 15 
train_matches = 15
eval_matches = 3
batch_size = 256
train_every_steps = 1
learning_rate = 1e-4
update_other_agent_every_steps = 1000 
memory_capacity = 3000 
update_target_model_every_steps = 500


# run_hyperparams = f"OPS={observations_per_state}_TM={train_matches}_EM={eval_matches}_BS={batch_size}_TES={train_every_steps}"
# run_name = "looooool"
# run_name = run_name + "---" + run_hyperparams
# run_path = f"runs/{run_name}"

eps_func = lambda x : max(0.1, 1. - x / 200000)
########################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = Environment(
  observations_per_state = observations_per_state
)

main_agent = DQNAgent(
  eps_func = eps_func,
  observations_per_state = observations_per_state,
  device = device,
  batch_size = batch_size,
  train_every_steps = train_every_steps,
  learning_rate = learning_rate,
  memory_capacity = memory_capacity,
  update_target_model_every_steps = update_target_model_every_steps
)

other_agent = DQNAgent(
  eps_func = eps_func,
  observations_per_state = observations_per_state,
  device = device,
  batch_size = batch_size,
  train_every_steps = train_every_steps,
  learning_rate = learning_rate,
  memory_capacity = memory_capacity,
  update_target_model_every_steps = update_target_model_every_steps
)

# run_name = "selfplay_target500_update1000_eps200000min0.1_lr1e-4_bs256_mem3000"
run_name = "updatedactions"
run_path = f"runs/{run_name}"
os.makedirs(run_path, exist_ok=True)
writer = SummaryWriter(run_path)

def train():
  main_agent.train_mode()
  # swapped = 0
  for _ in range(train_matches):
    state, info = env.reset(2)
    done = False
    while not done:
      action1 = main_agent.generate_action(state)
      action2 = other_agent.generate_action(state)
      # if swapped:
      #   action1, action2 = action2, action1
      # new_state, reward, done, info = env.step([action1, action2], player = swapped + 1)
      new_state, reward, done, info = env.step([action1, action2])
      start = time.time()
      loss = main_agent.train(state, action1, reward, new_state, done, info)
      print(f"Time for train: {(time.time() - start) * 1000} ms")
      if main_agent.step % update_other_agent_every_steps == 0:
        other_agent.model.load_state_dict(main_agent.model.state_dict())
      if loss >= 0:
        writer.add_scalar("loss", loss, main_agent.step)
      state = new_state
    # swapped ^= 1

def evaluate():
  main_agent.eval_mode()
  wins = 0
  rewards = 0
  dmg_dealt = 0
  dmg_taken = 0
  steps = 0

  with torch.no_grad():
    for _ in range(eval_matches):
      state, info = env.reset(1)
      done = False
      while not done:
        action = main_agent.generate_action(state)
        new_state, reward, done, info = env.step(action)
        state = new_state
        if "player1_hp_delta" in info:
          dmg_dealt += -info["player2_hp_delta"]
          dmg_taken += -info["player1_hp_delta"]
        rewards += reward
        steps += 1
      wins += info["winner"] == "player1"

  writer.add_scalar("winrate", wins / eval_matches * 100, main_agent.evaluations_done)
  writer.add_scalar("avg_dmg_dealt", dmg_dealt / eval_matches, main_agent.evaluations_done)
  writer.add_scalar("avg_dmg_taken", dmg_taken / eval_matches, main_agent.evaluations_done)
  writer.add_scalar("avg_reward", rewards / eval_matches, main_agent.evaluations_done)
  writer.add_scalar("avg_steps", steps / eval_matches, main_agent.evaluations_done)
  writer.flush()
  main_agent.evaluations_done += 1

model_path = f"{run_path}/save.lmao"
if os.path.exists(model_path):
  main_agent.load(model_path)

while True:
  main_agent.save(model_path)
  print(f"Training iteration {main_agent.evaluations_done} starting.")
  train()
  print(f"Training finished, evaluating agent.")
  evaluate()
