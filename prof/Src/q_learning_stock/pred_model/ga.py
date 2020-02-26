import numpy as np
import pandas as pd
import math
import calendar
import random
import indicators
import util

def cal_pop_fitness(population: np.ndarray, base_rates:list, df_list: list, trend_list: list, date_range: list, portfolio_comp: list):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = []
    for i in range(population.shape[0]):
        asset_list = [100000, 100000, 100000]
        util.cal_portfolio_comp_fitness(asset_list, base_rates, portfolio_comp, df_list, date_range, trend_list, 
            cvar_period=[population[i][0],population[i][1],population[i][2]], 
            mc_period=[population[i][3],population[i][4],population[i][5]], 
            sp_period=[population[i][6],population[i][7],population[i][8]],
            c1=[population[i][9],population[i][10],population[i][11]], 
            c2=[population[i][12],population[i][13],population[i][14]],
            thres= population[i][15], fitness=fitness)
    return np.array(fitness)

def select_mating_pool(population: np.ndarray, fitness: list, num_parents: int):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size) -> np.ndarray:
    offspring = np.empty(offspring_size)
    offspring_size[1]
    # The point at which crossover takes place between two parents. Usually it is at the center.
    random.seed()
    crossover_point1 = random.randrange(0, offspring_size[1]-1, 1)
    crossover_point2 = random.randrange(crossover_point1 + 1, offspring_size[1])

    # two point crossover
    for i in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = i % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (i+1) % parents.shape[0]
        # The new offspring will have its first part of its genes taken from the first parent.
        offspring[i, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
        # The new offspring will have its second part of its genes taken from the second parent.
        offspring[i, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
        # The new offspring will have its third part of its genes taken from the first parent.
        offspring[i, crossover_point2:] = parents[parent1_idx, crossover_point2:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for i in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = np.random.uniform(-1.0, 1.0, 2*3+1)
        random.seed()

        for j in range(3*3):
            new_sum = offspring_crossover[i, j] + random.randrange(-5, 5, 1)
            # Make sure period does not go to negative
            if not(new_sum < 2):
                offspring_crossover[i, j] = new_sum
        for j in range(3*3,5*3):
            offspring_crossover[i, j] = offspring_crossover[i, j] + random_value[j-3*3]
        new_sum = offspring_crossover[i, 5*3] + random_value[6] * 0.025
        if new_sum > 0 and new_sum < 0.1:
            offspring_crossover[i, 15] = new_sum
    return offspring_crossover
