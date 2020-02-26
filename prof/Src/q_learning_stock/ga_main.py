# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'eddy_src/q_learning_stock'))
	print(os.getcwd())
except:
	pass

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import pred_model
import config
import indicators
import util

#%% [markdown]
## Configure stock data for genetic algorithm. 
### Select stock set
#%%
run_set = ['goldman', 'index', '^BVSP', '^TWII', '^IXIC', 'index_sampled']
choose_set_num = 0
num_generations = 200

#%% [markdown]
## Genetic algorithm start
#%%
df_list, date_range, trend_list, _ = util.get_algo_dataset(choose_set_num)
base_rates = [0.2, 0.2, 0.2]

# number of parameters
num_weights = 5*3 + 1
sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop,num_weights)
periods_size = (sol_per_pop, 3*3)

new_population_periods = np.random.uniform(low=3, high=6, size=periods_size)
for row in new_population_periods:
    for i in range(len(row)):
        row[i] = round(row[i])
new_population_c = np.random.uniform(low=0, high=1, size=(sol_per_pop,2*3))
new_population_th = np.random.uniform(low=0.1, high=0.2, size=(sol_per_pop,1))
new_population = np.concatenate((new_population_periods, new_population_c, new_population_th), axis=1)

max_fitness = 0
portfolio_comp = [base_rates[i] + [0.4/3, 0.4/3, 0.4/3][i] for i in range(len(base_rates))]

for generation in tqdm(range(num_generations)):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = pred_model.cal_pop_fitness(new_population, base_rates, df_list, trend_list, date_range, portfolio_comp)
    print('Fitness for gen {}: {}'.format(generation, fitness))
    # The best result in the current iteration.
    print("Best result for generation {}: {}".format(generation, np.max(fitness)))
    max_fitness = max(max_fitness, np.max(fitness))
    print("Best result so far = {}".format(max_fitness))

    # Selecting the best parents in the population for mating.
    parents = pred_model.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = pred_model.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = pred_model.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    best_match_idx = np.where(fitness == np.max(fitness))
    sol_list = new_population[best_match_idx, :].flatten()
    best_sol = []
    for i in range(0,13,3):
        sol_group = []
        for j in range(3):
            sol_group.append(sol_list[i+j])
        best_sol.append(sol_group)
    best_sol.append(sol_list[15])
    print("Best solution so far", best_sol)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = pred_model.cal_pop_fitness(new_population, base_rates, df_list, trend_list, date_range, portfolio_comp)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
print(new_population[best_match_idx, :].shape)
sol_list = new_population[best_match_idx, :].flatten()
best_sol = []
for i in range(0,13,3):
    sol_group = []
    for j in range(3):
        sol_group.append(sol_list[i+j])
    best_sol.append(sol_group)
best_sol.append(sol_list[15])
print("Best solution fitness : ", best_sol)

#%%
