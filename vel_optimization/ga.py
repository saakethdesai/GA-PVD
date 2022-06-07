import random
from deap import base, creator, tools
import numpy as np
import time
from scipy.stats import truncnorm
from random import randint, gauss
import os
from sys import argv, path
from subprocess import call

import driver as driver

inputfile = "in.alloy_deposition"
nparams = 3 
ngens = 10
popsize = 40
selection_size = int(0.25*popsize)

np.random.seed(seed=123456)


def initialize_deap(gen):

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    a1 = 0.05; b1 = 1.00 #upper and lower bound for A
    a2 = 0.10; b2 = 0.90 #upper and lower bound for f
    a3 = 1; b3 = 100 #upper and lower bound for T

    bounds = [a1, b1, a2, b2, a3, b3]

    toolbox.register("create_individual", generate_individual, bounds=bounds, nparams=nparams)
    toolbox.register("population", tools.initRepeat, list, toolbox.create_individual)

    toolbox.register("mutate", custom_mutation, bounds=bounds, indpb=1.0)
    toolbox.register("select", tools.selBest, k=selection_size)

    pop = toolbox.population(n=popsize) #generate initial population
    if (gen > 0):
        filename = "gen" + str(gen-1) + "/" + "gen" + str(gen-1) + "_pop_info.txt"
        pop = read_population_state(filename, pop) #read in current state of population
    
    return pop, toolbox

def generate_individual(bounds, nparams):
   
    a1, b1, a2, b2, a3, b3 = bounds 
    ind = []
    #generate A
    A = np.random.uniform(a1, b1)
    #generate f 
    f = np.random.uniform(a2, b2)
    #generate period
    T = randint(a3, b3)*1000
    
    ind.append(A)
    ind.append(f)
    ind.append(T)
    ind_obj = creator.Individual(ind)
    
    return ind_obj


def read_population_state(filename, pop):
    raw_data = np.genfromtxt(filename)
    for idx, ind in enumerate(pop):
        raw_ind = raw_data[idx, 1:1+nparams]
        for i in range(len(ind)):
            ind[i] = raw_ind[i]
        if (np.isnan(raw_data[idx, -1])):
            pass 
        else:
            ind.fitness.values = (raw_data[idx, -1],)
    return pop


def custom_mutation(individual, bounds, indpb):
    
    for i in range(len(individual)):
        if (random.random() < indpb and i < nparams-1):
            current_state = individual[i]
            lower_end = max( 0.95*current_state, bounds[2*i] )
            upper_end = min( 1.05*current_state, bounds[2*i+1])
            individual[i] = np.random.uniform(lower_end, upper_end)
        elif (random.random() < indpb and i == nparams-1):
            current_state = individual[i]
            lower_end = max( int(0.95*current_state), bounds[2*i]*1000 )
            upper_end = min( int(1.05*current_state), bounds[2*i+1]*1000 )
            individual[i] = random.randint(lower_end, upper_end)

    return individual,


def get_bounded_params(params):
    
    input_params = np.array(params)
    A = input_params[0]
    f = input_params[1]
    T = input_params[2]
    a0 = A*f
    an = np.array([2*A/(n*np.pi) * np.sin(n*np.pi*f) for n in range(1, 6)])
    bn = np.array([0 for n in range(1, 6)])

    P = int(T)
    pi = np.pi
    it = np.arange(P)
    v = a0/2 + an[0]*np.cos(2*pi*it/P) + bn[0]*np.sin(2*pi*it/P) \
      + an[1]*np.cos(2*pi*it*2/P) + bn[1]*np.sin(2*pi*it*2/P) \
      + an[2]*np.cos(2*pi*it*3/P) + bn[2]*np.sin(2*pi*it*3/P) \
      + an[3]*np.cos(2*pi*it*4/P) + bn[3]*np.sin(2*pi*it*4/P) \
      + an[4]*np.cos(2*pi*it*5/P) + bn[4]*np.sin(2*pi*it*5/P)

    max_v = max(v) 
    min_v = min(v) 

    scaled_max = 0
    scaled_min = -max_v

    scaling_factor = (scaled_max - scaled_min)/(max_v - min_v)

    an_star = an*scaling_factor
    bn_star = bn*scaling_factor
    a0_star = a0*scaling_factor + 2*scaled_min - 2*min_v*scaling_factor

    a0_star = np.array([a0_star])
    P = np.array([P], dtype='int')
    bounded_params = list(np.concatenate([a0_star, an_star, bn_star, P]))

    return bounded_params

def run_simulations(gen, ind_set):
    
    for idx, ind in enumerate(ind_set):
        ind_name = "ind" + str(idx+1)
        params = ind
        os.mkdir(ind_name)
        os.chdir(ind_name)
        bounded_params = get_bounded_params(params)
        ind = bounded_params
        driver.write_input_file(inputfile, bounded_params)
        os.chdir("../")
    #submit job-array
    nruns = len(ind_set)
    job_array_id = driver.submit_job_array(gen, inputfile, nruns)

    return job_array_id


def eval_obj_func_prev_gen(gen, ind_list):

    os.chdir("gen"+str(gen-1))
    for idx, ind in enumerate(ind_list):
        ind_name = "ind" + str(idx+1)
        os.chdir(ind_name)
        path = os.getcwd()
        params = ind
        bounded_params = get_bounded_params(params)
        obj = driver.analyze_microstructure(path, bounded_params)
        ind.fitness.values = (obj,)
        os.chdir("../")

    return



def write_population_state(gen, pop, filename):
    f = open(filename, "w")
    for idx, ind in enumerate(pop):
        writestring = str(idx)
        writestring += "    " + " ".join(str(i) for i in ind)
        try:
            writestring += "    " + str(ind.fitness.values[0])
        except:
            writestring += "    " + str(np.nan)
        writestring += "\n"
        f.write(writestring)
    f.close()
    return



def launch_next_generation(gen, job_array_id):
    
    account = "fy190040"
    queue = "short,batch"
    jobname = "ga_gen" + str(gen)
    time = "03:00:00"

    nodes = str(1)
    tasks_per_node = str(1)
    
    header = ("#!/bin/bash\n"
              "#SBATCH --account=%s\n"
              "#SBATCH --partition=%s\n"
              "#SBATCH --job-name=%s\n"
              "#SBATCH --time=%s\n"
              "#SBATCH --nodes=%s\n"
              "#SBATCH --ntasks-per-node=%s\n") %(account, queue, jobname, time, nodes, tasks_per_node)
             
    variable_set = ("nodes=$SLURM_JOB_NUM_NODES\n"
                    "tasks=$SLURM_NTASKS_PER_NODE\n")

    module_loads = ("module load intel/19.0\n"
                    "module load mkl/19.0\n"
                    "module load openmpi-intel/3.1\n")
                    #"source activate neuroevolution\n")
                    #"module load anaconda3/5.2.0\n"

    run_command = ("python ga.py %s") %(gen)

    writeline = header + variable_set + module_loads + run_command

    f = open("run_ga.sh", "w")
    f.write(writeline)
    f.close()

    dependency_string = "--dependency=afterany:%s" %(job_array_id)
    call(["sbatch", dependency_string, "run_ga.sh"])

    return


def ga(gen):
    
    pop, toolbox = initialize_deap(gen)
    if (gen > 0):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        eval_obj_func_prev_gen(gen, invalid_ind)
        filename = "gen" + str(gen-1) + "_pop_info.txt"
        write_population_state(gen, pop, filename)
        os.chdir("../")
    if (gen == ngens):
        return
    
    gen_name = "gen" + str(gen)
    os.mkdir(gen_name)
    os.chdir(gen_name)
   
    if (gen == 0):
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    else:
        #clone the population
        offspring = list(map(toolbox.clone, pop))
        #select k best individuals
        selected_individuals = toolbox.select(offspring)
        #mutate best individuals to generate rest of population
        nmutations_per_ind = int((popsize - selection_size)/selection_size)
        offspring = [ind for ind in selected_individuals]
        for ind in selected_individuals:
            for i in range(nmutations_per_ind):
                cloned_ind = toolbox.clone(ind)
                toolbox.mutate(cloned_ind)
                offspring.append(cloned_ind)
                del cloned_ind.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        pop = offspring
   
    job_array_id = run_simulations(gen, invalid_ind)
    filename = "gen" + str(gen) + "_pop_info.txt"
    write_population_state(gen, pop, filename)
    
    os.chdir("../")
    
    launch_next_generation(gen+1, job_array_id)
    
    return 


if __name__ == "__main__":
    gen = int(argv[1])
    ga(gen)
