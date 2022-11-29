import time
import gym
import numpy as np
import pygad.torchga
import pygad
import torch
import torch.nn as nn
from multiprocessing import Pool


def get_gym():
    import flappy_bird_gym
    # env = gym.make("CartPole-v1")
    env = flappy_bird_gym.make("FlappyBird-v0")
    return env


STATE_DICT_PATH = './FlappyBird-v0.pth'
env = get_gym()

observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

model = nn.Sequential(
    nn.Linear(observation_space_size, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, action_space_size)
)
model.load_state_dict(torch.load(STATE_DICT_PATH))


# play game
with torch.no_grad():
    observation = env.reset()
    sum_reward = 0
    done = False
    while not done:
        env.render()
        ob_tensor = torch.tensor(observation.copy(), dtype=torch.float)
        q_values = model(ob_tensor)
        action = np.argmax(q_values).numpy()
        observation_next, reward, done, info = env.step(action)
        observation = observation_next
        sum_reward += reward
        # time.sleep(1/30)


print("Sum reward: " + str(sum_reward))
