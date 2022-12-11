import gym
import flappy_bird_gym
import gym_snake
 
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
    return env


if __name__ == "__main__":
    print(gym.envs.registry.all())
    env = make()
