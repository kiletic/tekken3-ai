from action import *
from enum import Enum
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class AgentModes(Enum):
  EVAL = 1
  TRAIN = 2

class Agent:
  def __init__(self, additional_forbidden_keys = [], additional_forbidden_actions = []):
    self.mode = AgentModes.TRAIN
    self.action_space = ActionSpace(additional_forbidden_keys, additional_forbidden_actions) 
    self.step = 0 # how many steps into the training
    self.evaluations_done = 0 # maybe this doesnt belong here...

  def eval_mode(self):
    self.mode = AgentModes.EVAL

  def train_mode(self):
    self.mode = AgentModes.TRAIN

  def generate_action(self, state):
    raise NotImplementedError("generate_action needs to be implemented in parent class")

  def save(self, model_path):
    raise NotImplementedError("save needs to be implemented")

  def load(self, model_path):
    raise NotImplementedError("load needs to be implemented")



class RandomAgent(Agent):
  def __init__(self, additional_forbidden_keys = [], additional_forbidden_actions = []):
    super().__init__(
      additional_forbidden_keys,
      additional_forbidden_actions
    )

  def generate_action(self, state):
    return self.action_space.generate_random_action()

  def train(self, *args):
    return -1

  def save(self, model_path):
    return 

  def load(self, model_path):
    return 

class DQNModel(nn.Module):
  def __init__(self, 
               in_channels, 
               num_of_actions,
               learning_rate = 5e-4):
    super(DQNModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    # 6 * 9 hardcoded for 102x76 image input
    self.fc1 = nn.Linear(6 * 9 * 64, 512)
    self.fc2 = nn.Linear(512, num_of_actions)
    self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate) 

  def forward(self, x):
    # print(f"x is {x.size()} as input")
    x = F.relu(self.conv1(x))
    # print(f"x is {x.size()} after conv1")
    x = F.relu(self.conv2(x))
    # print(f"x is {x.size()} after conv2")
    x = F.relu(self.conv3(x))
    # print(f"x is {x.size()} after conv3")
    x = F.relu(self.fc1(x.reshape(x.shape[0], -1)))
    # print(f"x is {x.size()} after fc1")
    x = self.fc2(x)
    # print(f"x is {x.size()} after fc2")
    return x 


class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen = capacity)

  def add(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DQNAgent(Agent):
  def __init__(self,
               eps_func,
               observations_per_state = 15, 
               device = "cpu", 
               batch_size = 16, 
               memory_capacity = 1600, 
               train_every_steps = 4, 
               mode = AgentModes.TRAIN,
               discount = 0.99,
               learning_rate = 5e-4,
               update_target_model_every_steps = 500,
               additional_forbidden_keys = [],
               additional_forbidden_actions = []):
    super().__init__(
      additional_forbidden_keys,
      additional_forbidden_actions
    )
    self.model = DQNModel(
      in_channels = observations_per_state, 
      num_of_actions = len(self.action_space), 
      learning_rate = learning_rate,
    ).to(device)
    self.target_model = DQNModel(
      in_channels = observations_per_state,
      num_of_actions = len(self.action_space),
      learning_rate = learning_rate,
    ).to(device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.memory = ReplayMemory(capacity = memory_capacity)
    self.device = device
    self.mode = mode 
    self.discount = discount
    self.train_every_steps = train_every_steps
    self.batch_size = batch_size
    self.eps_func = eps_func
    self.update_target_model_every_steps = update_target_model_every_steps 

    # this is needed just because of save/load
    self.memory_capacity = memory_capacity
    self.observations_per_state = observations_per_state

  def generate_action(self, state):
    eps = self.eps_func(self.step)
    if self.mode == AgentModes.EVAL or random.random() > eps:
      q_values = self.model(state.to(self.device).unsqueeze(0)).flatten() 
      action_idx = q_values.argmax().item()
      action = self.action_space.num_to_action[action_idx]
      print(f"{q_values} -> {action}, {action_idx}")
      return action 
    else:
      return self.action_space.generate_random_action() 

  def update(self, states, actions, rewards, new_states, dones):
    with torch.no_grad():
      max_next_q_values, _ = self.target_model(new_states).max(dim=1)
      y_j = rewards + self.discount * max_next_q_values * (~dones) 
    q_values = self.model(states).gather(1, actions.unsqueeze(0)).squeeze()

    loss = torch.nn.MSELoss()
    loss = loss(q_values, y_j)

    self.model.optimizer.zero_grad() 
    loss.backward()
    self.model.optimizer.step()

    return loss.item()

  def train(self, state, action, reward, new_state, done, info):
    state = state.to(self.device)
    new_state = new_state.to(self.device)
    self.memory.add((state, self.action_space.action_to_num[action], reward, new_state, done))
    self.step += 1

    if self.step % self.update_target_model_every_steps == 0:
      self.target_model.load_state_dict(self.model.state_dict())

    if len(self.memory) >= 2 * self.batch_size and self.step % self.train_every_steps == 0:
      minibatch = self.memory.sample(self.batch_size)
      states, actions, rewards, new_states, dones = zip(*minibatch)

      states = torch.stack(states)
      actions = torch.tensor(actions, device=self.device)
      rewards = torch.tensor(rewards, device=self.device)
      new_states = torch.stack(new_states) 
      dones = torch.tensor(dones, device=self.device)

      return self.update(states, actions, rewards, new_states, dones)
    else:
      return -1

  def save(self, model_path):
    print("Saving agent...")
    start = time.time()
    
    torch.save(
      {
        "model_state_dict" : self.model.state_dict(),
        "optimizer_state_dict" : self.model.optimizer.state_dict(),
        "agent_step" : self.step,
        "evaluations_done" : self.evaluations_done,
        "memory_capacity" : self.memory_capacity,
        "observations_per_state" : self.observations_per_state,
        "discount" : self.discount,
        "train_every_steps" : self.train_every_steps,
        "batch_size" : self.batch_size,
        "update_target_model_every_steps" : self.update_target_model_every_steps,
        "additional_forbidden_keys" : self.action_space.additional_forbidden_keys,
        "additional_forbidden_actions" : self.action_space.additional_forbidden_actions
      },
      model_path 
    )

    print(f"Saving done, it took {(time.time() - start) * 1000} ms")

  def load(self, model_path):
    print("Loading agent...")
    start = time.time()

    save_state = torch.load(model_path)

    if "additional_forbidden_keys" in save_state:
      self.action_space = ActionSpace(save_state["additional_forbidden_keys"], save_state["additional_forbidden_actions"])
      print(f"Loaded action space with {len(self.action_space)} actions.")

    self.observations_per_state = save_state["observations_per_state"]
    self.step = save_state["agent_step"]
    self.evaluations_done = save_state["evaluations_done"]
    self.memory_capacity = save_state["memory_capacity"]
    self.discount = save_state["discount"]
    self.train_every_steps = save_state["train_every_steps"]
    self.batch_size = save_state["batch_size"]
    self.update_target_model_every_steps = save_state["update_target_model_every_steps"]

    self.model = DQNModel(in_channels = self.observations_per_state, num_of_actions = len(self.action_space)).to(self.device)
    self.target_model = DQNModel(in_channels = self.observations_per_state, num_of_actions = len(self.action_space)).to(self.device)
    self.model.load_state_dict(save_state["model_state_dict"])
    self.target_model.load_state_dict(save_state["model_state_dict"])
    self.model.optimizer.load_state_dict(save_state["optimizer_state_dict"])

    self.memory = ReplayMemory(capacity = self.memory_capacity)

    
    print(f"Loading done, it took {(time.time() - start) * 1000} ms")

class DDQNAgent(DQNAgent):
  def __init__(self,
               eps_func,
               observations_per_state = 15, 
               device = "cpu", 
               batch_size = 16, 
               memory_capacity = 1600, 
               train_every_steps = 4, 
               mode = AgentModes.TRAIN,
               discount = 0.99,
               learning_rate = 5e-4,
               update_target_model_every_steps = 500,
               additional_forbidden_keys = [],
               additional_forbidden_actions = []):
    super().__init__(
      eps_func, 
      observations_per_state, 
      device, 
      batch_size,
      memory_capacity,
      train_every_steps,
      mode,
      discount,
      learning_rate,
      update_target_model_every_steps,
      additional_forbidden_keys,
      additional_forbidden_actions
    )

  def update(self, states, actions, rewards, new_states, dones):
    _, chosen_actions = self.model(new_states).max(dim=1)
    with torch.no_grad():
      target_q_vals = self.target_model(new_states).gather(1, chosen_actions.unsqueeze(0)).squeeze()
      y_j = rewards + self.discount * target_q_vals * (~dones)
    q_values = self.model(states).gather(1, actions.unsqueeze(0)).squeeze()

    loss = torch.nn.MSELoss()
    loss = loss(q_values, y_j)

    self.model.optimizer.zero_grad() 
    loss.backward()
    self.model.optimizer.step()

    return loss.item()
