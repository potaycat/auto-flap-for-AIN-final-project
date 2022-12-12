import time
import numpy as np
import torch
import get_gym_and_model


env, model, STATE_DICT_PATH = get_gym_and_model.make()
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

        observation, reward, done, info = env.step(action)
        sum_reward += reward
        time.sleep(1 / 30)


print("Sum reward: " + str(sum_reward))
