# Tekken 3 AI
This project uses reinforcement learning algorithm to train an agent to play the video game Tekken 3. It was made as part of my master's thesis.

## How does it work?
To apply reinforcement learning algorithms, a method of communication between reinforcement learning agent and the video game needed to be built. [PCSX-Redux](https://github.com/grumpycoders/pcsx-redux) was used to emulate the game, and [PyTorch](https://github.com/pytorch/pytorch) was used as a base for reinforcement learning algorithms.

### Communication

Communication between the emulator and python program was achieved through TCP sockets. One instance of the emulator connects to the agent and then communication starts. Agent receives all information it needs from the game by requests which are sent through TCP. Requests are received in Lua environment that is part of Redux, and uses its [Lua API](https://pcsx-redux.consoledev.net/Lua/introduction/) to control the game or read information which is then sent back to the agent if needed. 

To handle network, [socket](https://docs.python.org/3/library/socket.html) library was used in python and [File API](https://pcsx-redux.consoledev.net/Lua/file-api/#network-streams) was used in Redux which in the background uses [libuv](https://pcsx-redux.consoledev.net/Lua/libraries/#luv).

<p align="center" width="100%">
    <img src="https://i.imgur.com/IGL55TZ.png">
</p>


### Agent
All implementations of the algorithms can be found in [agent.py](https://github.com/kiletic/tekken3-ai/blob/main/python/agent.py).

To train the agent deep reinforcement learning was used, specifically implementation of the **Deep Q-Learning** algorithm as described in [[1]](https://www.nature.com/articles/nature14236), and **Double Deep Q-Learning** as described in [[2]](https://arxiv.org/abs/1509.06461). Following image shows simplified view of the components implemented.

<p align="center" width="100%">
    <img src="https://i.imgur.com/1iQJ19e.png">
</p>

State is represented as **n** consecutive frames in the game, grayscaled and resized to 102x72. That is then turned into PyTorch tensor and fed as input to the agent.

<p align="center" width="100%">
    <img src="https://imageupload.io/ib/ziUtAD1RxasT94f_1697738932.png">
</p>

## Example
As an example of the most basic usage following code is provided. Here, we let an agent that just sends random moves play 100 rounds against the CPU AI (opponent provided by the game). Syntax was heavily inspired by [OpenAI's Gym](https://github.com/openai/gym) project.

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

**Note**: although we call `agent.train(...)` nothing actually happens inside since a RandomAgent doesn't learn

## Result

After training for 1.15M steps which correspond to 32 hours or 17250 matches played, agent achieved \~90% winrate against CPU AI on easy difficulty and \~33% winrate on same opponent but hard difficulty. Looking at the graphs and comparing it to RandomAgent which achieves \~79%/\~25% it is clear that the agent is successfuly learning. 

<p align="center" width="100%">
    <img width="60%" src="https://i.imgur.com/WkKH8Kg.png">
</p>


## References
[1] Mnih, V., Kavukcuoglu, K., Silver, D. et al., Human-level control through deep
reinforcement learning, Nature 518, 529–533 (2015)

[2] H. V. Hasselt, A. Guez, D. Silver, Deep Reinforcement Learning with Double
Q-learning, arxiv:1509.06461 (2015)
