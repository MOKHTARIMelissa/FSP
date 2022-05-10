# -*-coding:utf-8 -*
import os
import random
import time
from functools import reduce
import numpy as np
import FSP as fsp


def calc_makespan(solution, proccessing_time, number_of_jobs, number_of_machines):
    # list for the time passed until the finishing of the job
    cost = [0] * number_of_jobs
    # for each machine, total time passed will be updated
    for machine_no in range(0, number_of_machines):
        for slot in range(number_of_jobs):
            # time passed so far until the task starts to process
            cost_so_far = cost[slot]
            if slot > 0:
                cost_so_far = max(cost[slot - 1], cost[slot])
            cost[slot] = cost_so_far + proccessing_time[solution[slot]][machine_no]
    return cost[number_of_jobs - 1]

    # La fonction objective à minimiser - Programmation dynamique (On suppose la numérotation commence de 1)
def calc_makespan_an(seq, processing_time,N,M):
        dt=np.array(processing_time)
        data=dt.T
        cmax = np.zeros(data.shape)
        sequence=np.array(seq)
        num=int(sequence[0]) -1
        info=data[0][num]
        cmax[0][0] = info

        for i in range(1, M):
            cmax[i,0] = cmax[i-1,0] + data[i][int(sequence[0]) -1]

        for i in range(1,len(sequence)):
            cmax[0,i] = cmax[0,i-1] + data[0][int(sequence[i]) -1]

        for i in range(1, M):
            for j in range(1,len(sequence)):
                cmax[i,j] = max(cmax[i,j-1], cmax[i-1,j]) + data[i][int(sequence[j]) -1]

        return cmax[M-1, len(sequence)-1]

def initialize_population(population_size, number_of_jobs,mode="random"):
    if mode=="random" :
        population = []
        i = 0
        while i < population_size:
            individual = list(np.random.permutation(number_of_jobs))
            if individual not in population:
                population.append(individual)
                i += 1
    elif mode=="":
        population=[]

    return population

# Two-point crossover is that the set of jobs between 
# two randomly selected points is always inherited from one parent to the child, 
# and the other jobs are placed in the same manner as the one-point crossover. 
def crossover(parents):
    parent1 = parents[0]
    parent2 = parents[1]
    length_of_parent = len(parent1)
    first_point = int(length_of_parent / 2 - length_of_parent / 4)
    second_point = int(length_of_parent - first_point)
    intersect = parent1[first_point:second_point]

    child = []
    index = 0
    for pos2 in range(len(parent2)):
        if first_point <= index < second_point:
            child.extend(intersect)
            index = second_point
        if parent2[pos2] not in intersect:
                child.append(parent2[pos2])
                index += 1

    return child

# apply mutation to an existing solution using swap move operator
def mutation(solution):
    # copy the solution
    mutated_solution = list(solution)
    solution_length = len(solution)
    # pick 2 positions to swap randomly
    swap_positions = list(np.random.permutation(np.arange(solution_length))[:2])
    first_job = solution[swap_positions[0]]
    second_job = solution[swap_positions[1]]
    mutated_solution[swap_positions[0]] = second_job
    mutated_solution[swap_positions[1]] = first_job
    return mutated_solution

# Selects parent by binary tournament method
def select_parent(fsp,population, processing_time, number_of_jobs, number_of_machines,inter_population_number,mode="tournois"):
    if mode=="tournois":
        parent_pairs = []
        # randomly choose how many parent pairs will be selected
        parent_pair_count = random.randint(2, int(len(population)/2))
        for k in range(parent_pair_count):
            parent1 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
            parent2 = binary_tournament(number_of_jobs, number_of_machines, population, processing_time)
            if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
                parent_pairs.append((parent1, parent2))
        return parent_pairs
    elif mode=="random":
        parent_pairs = []
        # randomly choose how many parent pairs will be selected
        parent_pair_count = random.randint(2, int(len(population)/2))
        for k in range(parent_pair_count):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
                parent_pairs.append((parent1, parent2))
        return parent_pairs
    elif mode=="elit":
        parent_pairs=[]
        parent_pair_count = random.randint(2, int(len(population)/2))
        pop_cmax=[fsp.Cmax(number_of_machines,indiv) for indiv in population]
        ordre=np.argsort(pop_cmax)
        pop=[population[pos] for pos in ordre[:inter_population_number] ]
        
        for k in range(parent_pair_count):
            parent1 = random.choice(pop)
            parent2 = random.choice(pop)
            if parent1 != parent2 and (parent1, parent2) not in parent_pairs:
                parent_pairs.append((parent1, parent2))
                      
    elif mode=="roulette":
        parent_pairs=[]
        parent_pair_count = random.randint(2, int(len(population)/2))
        pop_cmax=[Cmax(number_of_machines,indiv) for indiv in population]   
        pop_cmax=np.max(pop_cmax)-pop_cmax+1
        parent_pairs=[random.choice(sequence=population, weights=pop_cmax,k=2) for pos in range(parent_pair_count)]
                      
    elif mode=="ranking":
        parent_pairs=[]
        parent_pair_count = random.randint(2, int(len(population)/2))
        pop_cmax=[Cmax(number_of_machines,indiv) for indiv in population]   
        ordre=np.argsort(pop_cmax)
        ordre=np.flip(ordre)
        pop=[population[pos] for pos in ordre[:inter_population_number]]
        parent_pairs=[random.choice(sequence=pop, weights=(ordre[:inter_population_number]+1),k=2) for pos in range(parent_pair_count)]
        
    return parent_pairs        

def binary_tournament(number_of_jobs, number_of_machines, population, processing_time):
    parent = []
    candidates = random.sample(population, 2)
    makespan1 = calc_makespan(candidates[0], processing_time, number_of_jobs, number_of_machines)
    makespan2 = calc_makespan(candidates[1], processing_time, number_of_jobs, number_of_machines)
    if makespan1 < makespan2:
        parent = candidates[0]
    else:
        parent = candidates[1]
    return parent

def update_population(population, children,processing_time,no_of_jobs,no_of_machines):
    costed_population = []
    for individual in population:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_population.append(ind_makespan)
    costed_population.sort(key=lambda x: x[0], reverse=True)

    costed_children = []
    for individual in children:
        ind_makespan = (calc_makespan(individual, processing_time, no_of_jobs, no_of_machines), individual)
        costed_children.append(ind_makespan)
    costed_children.sort(key=lambda x: x[0])
    for child in costed_children:
        if child not in population:
            population.append(individual)
            population.remove(costed_population[0][1])
            break