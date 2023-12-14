# Tekken 3 AI
This project uses a reinforcement learning algorithm to train an agent to play the video game Tekken 3. It was made as part of my master's thesis.

## How does it work?
To apply reinforcement learning algorithms, a method of communication between the reinforcement learning agent and the video game needed to be established. [PCSX-Redux](https://github.com/grumpycoders/pcsx-redux) was used to emulate the game, and [PyTorch](https://github.com/pytorch/pytorch) served as the foundation for the reinforcement learning algorithms. All code written for emulator side is located in the [lua/](https://github.com/kiletic/tekken3-ai/tree/main/lua) folder and all code for reinforcement learning can be found in the [python/](https://github.com/kiletic/tekken3-ai/tree/main/python) folder.

### Communication

Communication between the emulator and Python program was achieved through TCP sockets. One instance of the emulator connects to the agent, and then communication begins. The agent receives all the information it needs from the game through requests sent via TCP. Requests are received in the Lua environment, which is a part of Redux, and it utilizes its [Lua API](https://pcsx-redux.consoledev.net/Lua/introduction/) to control the game or read information, which is then sent back to the agent if needed. 

To handle network, [socket](https://docs.python.org/3/library/socket.html) library was used in Python and [File API](https://pcsx-redux.consoledev.net/Lua/file-api/#network-streams) was used in Redux which in the background utilizes [libuv](https://pcsx-redux.consoledev.net/Lua/libraries/#luv).

<p align="center" width="100%">
    <img src="https://i.imgur.com/PSb5Us5.png">
</p>


### Agent
All implementations of the algorithms can be found in [agent.py](https://github.com/kiletic/tekken3-ai/blob/main/python/agent.py).

To train the agent deep reinforcement learning was used, specifically implementation of the **Deep Q-Learning** algorithm as described in [[1]](https://www.nature.com/articles/nature14236), and **Double Deep Q-Learning** as described in [[2]](https://arxiv.org/abs/1509.06461). The following image shows a simplified view of the components implemented.

<p align="center" width="100%">
    <img src="https://i.imgur.com/q3UiqEq.png">
</p>

The state is represented as **n** consecutive frames in the game, grayscaled and resized to 102x72. These are then converted into PyTorch tensors and fed as input to the agent.

<p align="center" width="100%">
    <img src="https://iili.io/JuRhzVn.md.png">
</p>

## Example
As an example of the most basic usage following code is provided. Here, we let an agent that just sends random moves play 100 rounds against the CPU AI (the opponent provided by the game). The syntax was heavily inspired by [OpenAI's Gym](https://github.com/openai/gym) project.

```py
from env import Environment
from agent import RandomAgent

env = Environment()
agent = RandomAgent()

for _ in range(100):
  state, info = env.reset(1)
  done = False
  while not done:
    action = agent.generate_action(state)
    new_state, reward, done, info = env.step(action)
    loss = agent.train(state, action, reward, new_state, done, info)
    state = new_state
```

**Note**: Although we call `agent.train(...)`, nothing actually happens inside since a RandomAgent doesn't learn.

<br/>

This API offers three most important functions:
- `env.reset(starting_state_index)` - resets the game to a predefined starting state using the emulator's load state functionality
- `env.step(action)` - sends an action to the emulator, which is then executed in the game, and returns the next state, the reward associated with the sent action, an indicator whether the match is finished after the action was executed (done), and some auxiliary info which is meant for human consumption (for example, if the game is paused, who is the winner, etc.)
- `agent.train(state, action, reward, new_state, done, info)` - one iteration of the reinforcement learning algorithm on a new experience `(state, action, new_state, reward)`  

## Result

The agent was playing as the fighter King, and the CPU AI was playing as Law. After training for 1.15M steps, which correspond to 96 hours of human play or 17250 played matches, the agent achieved a \~90% winrate against the CPU AI on easy difficulty and a \~33% winrate on hard difficulty. Looking at the graphs and comparing it to the RandomAgent, which achieves \~79%/\~25%, it is clear that the agent is successfuly learning. 

<p align="center" width="100%">
    <img width="60%" src="https://i.imgur.com/Wfqs36N.png">
</p>


## References
[1] Mnih, V., Kavukcuoglu, K., Silver, D. et al., Human-level control through deep
reinforcement learning, Nature 518, 529–533 (2015)

[2] H. V. Hasselt, A. Guez, D. Silver, Deep Reinforcement Learning with Double
Q-learning, arxiv:1509.06461 (2015)
