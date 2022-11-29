import time
import numpy as np
import pygad.torchga
import pygad
import torch
import torch.nn as nn
from multiprocessing import Pool
import get_gym


def fitness_func(solution, sol_idx):
    global model, observation_space_size, env

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False
    while (not done) and (sum_reward < 1000):
        # env.render()
        ob_tensor = torch.tensor(observation.copy(), dtype=torch.float)
        q_values = model(ob_tensor)
        action = np.argmax(q_values).numpy()
        observation_next, reward, done, info = env.step(action)
        observation = observation_next
        sum_reward += reward

    return sum_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))


def fitness_wrapper(solution):
    return fitness_func(solution, 0)


class PooledGA(pygad.GA):
    def cal_pop_fitness(self):
        pop_fitness = self.pool.map(fitness_wrapper, self.population)
        print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

env = get_gym.make()
observation_space_size = env.observation_space.shape[0]

action_space_size = env.action_space.n

torch.set_grad_enabled(False)

model = nn.Sequential(
    nn.Linear(observation_space_size, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, action_space_size)
)


if __name__ == '__main__':
    start_time = time.time()

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=10)

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    parameters = {
        'num_generations': 50,
        'num_parents_mating': 5,
        'initial_population': torch_ga.population_weights,
        'fitness_func': fitness_func,
        'parent_selection_type': "sss",
        'crossover_type': "single_point",
        'mutation_type': "random",
        'mutation_percent_genes': 10,
        'keep_parents': -1,
        'on_generation': callback_generation,
    }
    ga_instance = pygad.GA(**parameters)
    ga_instance.run()

    # ga_instance = PooledGA(**parameters)
    with Pool(processes=10) as pool:
        ga_instance.pool = pool
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    print(f"--- {time.time() - start_time} seconds ---")

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)
    torch.save(model.state_dict(),
               f"./saved_model/{env.unwrapped.spec.id}.pth")

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(
        title="PyGAD & Dense NN - Iteration vs. Fitness", linewidth=4)
