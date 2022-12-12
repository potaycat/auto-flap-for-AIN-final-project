import gym
import flappy_bird_gym
import gym_snake
import torch.nn as nn

"""
LIST OF TESTED GYMS:
 - CartPole-v1
 - FlappyBird-v0
 - Snake-v0
"""


def make():
    env = gym.make("FlappyBird-v0")
    # env.seed()
    # env.set_rewards(rew_step=-4.9, rew_apple=50, rew_death=-30, rew_death2=-50)

    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    model = nn.Sequential(
        nn.Linear(observation_space_size, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, action_space_size),
    )
    STATE_DICT_PATH = f"./saved_models/{env.unwrapped.spec.id}.pth"

    return env, model, STATE_DICT_PATH


if __name__ == "__main__":
    print(gym.envs.registry.all())
    make()
