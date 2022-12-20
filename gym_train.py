import time
import numpy as np
import pygad.torchga
import pygad
import torch
from multiprocessing import Pool
import get_gym_and_model


def fitness_func(solution, sol_idx):
    global model, env

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution
    )
    model.load_state_dict(model_weights_dict)

    # play game
    observation = env.reset()
    sum_reward = 0
    done = False
    while (not done) and (-100 < sum_reward < 100000):
        # env.render()
        ob_tensor = torch.tensor(observation.copy(), dtype=torch.float)
        q_values = model(ob_tensor)
        action = np.argmax(q_values).numpy()
        observation, reward, done, info = env.step(action)
        sum_reward += reward

    return sum_reward


def callback_generation(ga_instance):
    print("Generation:", ga_instance.generations_completed, end=" - ")
    print("Fitness:", ga_instance.best_solution()[1])


def fitness_wrapper(solution):
    return fitness_func(solution, 0)


class PooledGA(pygad.GA):
    def cal_pop_fitness(self):
        pop_fitness = self.pool.map(fitness_wrapper, self.population)
        # print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness


torch.set_grad_enabled(False)

env, model, STATE_DICT_PATH = get_gym_and_model.make()
# model.load_state_dict(torch.load(STATE_DICT_PATH)) # resume training

if __name__ == "__main__":
    start_time = time.time()

    torch_ga = pygad.torchga.TorchGA(model=model, num_solutions=100)

    # https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
    parameters = {
        "num_generations": 200,
        "num_parents_mating": 50,
        "initial_population": torch_ga.population_weights,
        "fitness_func": fitness_func,
        "parent_selection_type": "sss",
        "crossover_type": "single_point",
        "mutation_type": "random",
        "mutation_percent_genes": 30,
        "keep_parents": -1,
        "on_generation": callback_generation,
    }
    # ga_instance = pygad.GA(**parameters)
    ga_instance = PooledGA(**parameters)
    with Pool(processes=50) as pool:
        ga_instance.pool = pool
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    print(f"--- {time.time() - start_time} seconds ---")

    model_weights_dict = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution
    )
    model.load_state_dict(model_weights_dict)
    torch.save(model.state_dict(), STATE_DICT_PATH)

    # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
    ga_instance.plot_fitness(
        title=f"PyGAD & Dense NN - {env.unwrapped.spec.id} Fitness", linewidth=4
    )
