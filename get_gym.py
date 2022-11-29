import gym
import flappy_bird_gym

def make():
    # env = gym.make("CartPole-v1")
    env = flappy_bird_gym.make("FlappyBird-v0")
    return env