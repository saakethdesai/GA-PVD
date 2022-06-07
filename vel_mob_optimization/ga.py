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
nparams = 13 
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
    a4 = 0.10; b4 = 5.00 #upper and lower bounds for M_phi
    a5 = 5.00; b5 = 25.00 #upper and lower bounds for M_surf_c
    a6 = 0.10; b6 = 5.00 #upper and lower bounds for M_bulk_c
    a7 = 0.001; b7 = 0.100 #upper and lower bounds for D
    a8 = 0.10; b8 = 1.00 #upper and lower bounds for switch_time
    a9 = 0; b9 = 1 #0 = lower; 1 = upper
    
    bounds = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, a8, b8, a9, b9]

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
   
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, a8, b8, a9, b9 = bounds
    ind = []
    #generate A
    A = np.random.uniform(a1, b1)
    #generate f 
    f = np.random.uniform(a2, b2)
    #generate period
    T = randint(a3, b3)*1000
    
    #generate direction of change in step 
    direction_change = np.random.choice([a9, b9])
    
    #1.05 and 0.95 so that there is room to go up or down
    #generate M_phi_i, M_phi_f
    M_phi_i = np.random.uniform(1.05*a4, 0.95*b4)
    M_phi_f = np.random.uniform(a4, M_phi_i) if (direction_change == 0) else np.random.uniform(M_phi_i, b4)
    #generate M_surf_c_i, M_surf_c_f
    M_surf_c_i = np.random.uniform(1.05*a5, 0.95*b5)
    M_surf_c_f = np.random.uniform(a5, M_surf_c_i) if (direction_change == 0) else np.random.uniform(M_surf_c_i, b5)
    #generate M_bulk_c_i, M_bulk_c_f
    M_bulk_c_i = np.random.uniform(1.05*a6, 0.95*b6)
    M_bulk_c_f = np.random.uniform(a6, M_bulk_c_i) if (direction_change == 0) else np.random.uniform(M_bulk_c_i, b6)
    #generate D_i, D_f 
    D_i = np.random.uniform(1.05*a7, 0.95*b7)
    D_f = np.random.uniform(a7, D_i) if (direction_change == 0) else np.random.uniform(D_i, b7)
    #generate ramp start, ramp end 
    switch_time = np.random.uniform(a8, b8)

    gene_list = [A, f, T, M_phi_i, M_phi_f, M_surf_c_i, M_surf_c_f, M_bulk_c_i, M_bulk_c_f, D_i, D_f, switch_time, direction_change] 
    ind.extend(gene_list)
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
    
    for i in range(len(individual)-1):
        if (random.random() < indpb and i < 2):
            current_state = individual[i]
            lower_end = max( 0.95*current_state, bounds[2*i] )
            upper_end = min( 1.05*current_state, bounds[2*i+1])
            individual[i] = np.random.uniform(lower_end, upper_end)
        elif (random.random() < indpb and i == 2):
            current_state = individual[i]
            lower_end = max( int(0.95*current_state), bounds[2*i]*1000 )
            upper_end = min( int(1.05*current_state), bounds[2*i+1]*1000 )
            individual[i] = random.randint(lower_end, upper_end)
        elif (random.random() < indpb and i > 2):
            current_state = individual[i]
            if (i % 2 == 0):
                lower_end = max( int(0.95*current_state), bounds[i+2] )
                upper_end = min( int(1.05*current_state), bounds[i+3] )
            elif (i % 2 == 1):
                lower_end = max( int(0.70*current_state), bounds[i+3] )
                upper_end = min( int(1.30*current_state), bounds[i+4] )
            individual[i] = np.random.uniform(lower_end, upper_end)
    individual[len(individual)-1] = np.random.choice([0, 1]) 

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
        params = ind[:3]
        temp_params = ind[3:]
        os.mkdir(ind_name)
        os.chdir(ind_name)
        bounded_params = get_bounded_params(params)
        ind = bounded_params
        driver.write_input_file(inputfile, bounded_params, temp_params)
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
        params = ind[:3]
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
    
    account = "FY220020"
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
