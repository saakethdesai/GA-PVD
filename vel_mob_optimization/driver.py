import numpy as np
from subprocess import call, check_output
from os import getenv, getcwd
from sys import path

import paraview.simple as pv
from glob import glob

from sklearn.decomposition import PCA
import pickle as pkl

from pymks import PrimitiveBasis
from pymks.stats import autocorrelate


def write_input_file(inputfile, params, temp_params):

    header = ("! *** Phase-field model settings\n"
              "phase-field_model                 alloy_deposition\n")
   
    film_params = ("initial_film-height               10.0d0\n"
                   "surface_roughness                 0.0d0 0.00d0 0.0d0\n"
                   "composition_distribution          0.0d0 0.35d0 -1.0d0 1.0d0\n"
                   "growth-restriction_value          0.75d0\n"
                   "growth-limit_value                0.35d0\n")
    
    M_phi_i = str(temp_params[0]) + "d0"
    M_phi_f = str(temp_params[1]) + "d0"
    M_phi_switch_time = str(temp_params[8]) + "d0"
    M_surf_c_i = str(temp_params[2]) + "d0"
    M_surf_c_f = str(temp_params[3]) + "d0"
    M_surf_c_switch_time = str(temp_params[8]) + "d0"
    M_bulk_c_i = str(temp_params[4]) + "d0"
    M_bulk_c_f = str(temp_params[5]) + "d0"
    M_bulk_c_switch_time = str(temp_params[8]) + "d0"

    free_energy_params = ("mobility_coefficient_phi_function step\n"
                          "initial_mobility_coefficient_phi  %s\n"
                          "final_mobility_coefficient_phi    %s\n"
                          "mobility_coefficient_phi_switch_time_fraction          %s\n"
                          "free-energy_barrier_phi           1.0d0\n"
                          "gradient-energy_coefficient_phi   1.0d0\n"
                          "surface-localization_chi          0.2d0\n"
                          "surface_mobilities_chi_function   step\n" 
                          "initial_surface_mobilities_chi    %s\n"
                          "final_surface_mobilities_chi      %s\n"
                          "surface_mobilities_chi_switch_time_fraction            %s\n"
                          "mobility_coefficients_chi_function step\n"
                          "initial_mobility_coefficients_chi %s\n"
                          "final_mobility_coefficients_chi   %s\n"
                          "mobility_coefficients_chi_switch_time_fraction          %s\n"
                          "free-energy_barrier_chi           1.0d0\n"
                          "gradient-energy_coefficient_chi   1.0d0\n") %(M_phi_i, M_phi_f, M_phi_switch_time, 
                                                                         M_surf_c_i, M_surf_c_f, M_surf_c_switch_time, 
                                                                         M_bulk_c_i, M_bulk_c_f, M_bulk_c_switch_time)

    a = params[0:6]
    b = params[6:11]
    P = int(params[11])

    an = a[1:]
    bn = b
    a0 = a[0]
    it = np.arange(P)
    pi = np.pi
    v = a0 + an[0]*np.cos(2*pi*it/P) + bn[0]*np.sin(2*pi*it/P) \
      + an[1]*np.cos(2*pi*it*2/P) + bn[1]*np.sin(2*pi*it*2/P) \
      + an[2]*np.cos(2*pi*it*3/P) + bn[2]*np.sin(2*pi*it*3/P) \
      + an[3]*np.cos(2*pi*it*4/P) + bn[3]*np.sin(2*pi*it*4/P) \
      + an[4]*np.cos(2*pi*it*5/P) + bn[4]*np.sin(2*pi*it*5/P)
    
    max_series = min(v)
    sim_time = int(np.abs(0.5*512/(max_series*1e-3) * 7)) 

    a = [str(i) + "d0" for i in a]
    a = ' '.join(a)
    b = [str(i) + "d0" for i in b]
    b = ' '.join(b)

    D_i = str(temp_params[6]) + "d0"
    D_f = str(temp_params[7]) + "d0"
    D_switch_time = str(temp_params[8]) + "d0"

    deposition_params  = ("vapor_diffusivity_function        step\n"
                          "initial_vapor_diffusivity         %s\n"
                          "final_vapor_diffusivity           %s\n"
                          "vapor_diffusivity_switch_time_fraction          %s\n"
                          "initial_mass-density              1.0d0\n"
                          "velocity_function                 series\n"
                          "velocity-field_a_components       %s\n"
                          "velocity-field_b_components       %s\n"
                          "velocity_period                   %s\n") %(D_i, D_f, D_switch_time, a, b, str(P))


    output_params = ("output_diary-data                 logfreq1 100 9 10\n"
                     "output_field-data                 serial vtk binary uniform 100000\n"
                     "! *** Numerical methods settings\n"
                     "time-step_size                    1.0d-3\n"
                     "time-step_iterations              %s\n"
                     "domain-size_physical              512 512 1\n"
                     "domain-size_numerical             512 512 1\n"
                     "vapor_field                       midpoint_method   p  0.0d0  p  0.0d0   d  0.0d0  d  1.0d0   p  0.0d0  p  0.0d0\n"
                     "solid_field                       midpoint_method   p  0.0d0  p  0.0d0   d  1.0d0  d -1.0d0   p  0.0d0  p  0.0d0\n"
                     "composition_field                 midpoint_method   p  0.0d0  p  0.0d0   n  0.0d0  n  0.0d0   p  0.0d0  p  0.0d0\n") %(sim_time)

    writeline = header + film_params + free_energy_params + deposition_params + output_params 

    f = open(inputfile, "w")
    f.write(writeline)
    f.close()

    return



def submit_job_array(gen, inputfile, nruns):
    
    account = "FY220020"
    queue = "batch"
    jobname = "job_array_gen_" + str(gen)
    time = "08:00:00"

    nodes = str(4)
    tasks_per_node = str(32)
    
    memphis_path = "/ascldap/users/saadesa/Softwares/MEMPHIS/memphis_modified_both/source/memphis.x" 

    array_string = str(1) + "-" + str(nruns)
    header = ("#!/bin/bash\n"
              "#SBATCH --account=%s\n"
              "#SBATCH --partition=%s\n"
              "#SBATCH --job-name=%s\n"
              "#SBATCH --time=%s\n"
              "#SBATCH --nodes=%s\n"
              "#SBATCH --ntasks-per-node=%s\n"
              "#SBATCH --array=%s\n") %(account, queue, jobname, time, nodes, tasks_per_node, array_string)
             
    variable_set = ("nodes=$SLURM_JOB_NUM_NODES\n"
                    "tasks=$SLURM_NTASKS_PER_NODE\n")

    module_loads = ("module load intel/19.0\n"
                    "module load mkl/19.0\n"
                    "module load openmpi-intel/3.1\n")
                    #"source activate neuroevolution\n")
                    #"module load anaconda3/5.2.0\n"

    dirname = "ind" + "${SLURM_ARRAY_TASK_ID}"
    dir_change = ("cd %s\n") %(dirname)
    run_command = ("mpirun --bind-to core --npernode $tasks --n $(($nodes*$tasks)) %s %s\n") %(memphis_path, inputfile)
    dir_change_back = ("cd ../\n")

    writeline = header + variable_set + module_loads + dir_change + run_command + dir_change_back

    f = open("run_array.sh", "w")
    f.write(writeline)
    f.close()

    slurm_output = check_output(['sbatch', 'run_array.sh'])
    slurm_output = slurm_output.decode("utf-8")
    job_array_id = int(slurm_output.split()[-1])
    
    return job_array_id


def analyze_microstructure(path, params):
    
    #search for last VTK file in path if multiple, else pick the VTK file
    composition_file_name = check_output("ls -v1 -r out.pvd_c*.vtk | head -n1", shell=True).decode("utf-8").rstrip("\n")
    solid_file_name = check_output("ls -v1 -r out.pvd_s*.vtk | head -n1", shell=True).decode("utf-8").rstrip("\n")

    #convert VTK to CSV using paraview LegacyVTKReader
    outpvd_c_vtk = pv.LegacyVTKReader(FileNames=[composition_file_name])
    outpvd_s_vtk = pv.LegacyVTKReader(FileNames=[solid_file_name])

    pv.SaveData("phase_info.csv", proxy=outpvd_c_vtk)
    pv.SaveData("solid_info.csv", proxy=outpvd_s_vtk)

    #Read data from CSV
    phase = np.genfromtxt("phase_info.csv", delimiter=",", skip_header=1)
    solid = np.genfromtxt("solid_info.csv", delimiter=",", skip_header=1)
    
    phase = phase[solid > 0] #work only with the solid phase

    phase[phase<0]=0
    phase[phase>0]=1

    L = len(phase)
    width = 512
    height = int(L/512)
    
    if (height >= 150):
        phase = phase[:width*height] #convert to (*, 512) [* = height of solid film]

        phase = phase.reshape((height, width))
        phase = np.flip(phase, 0)
        phase = phase.reshape((1, height, width))
        phase = phase[:, :150, :]
        
        basis = PrimitiveBasis(n_states=2)
        correlation = autocorrelate(phase, basis, periodic_axes=(1,))
        correlation = correlation[..., 0]
        flatten = correlation.reshape((correlation.shape[0], correlation.shape[1]*correlation.shape[2]))
        pca_model = pkl.load(open("/qscratch/saadesa/ml_thin_films/pca_training_set/train_models/simulations_only/15/model.pkl", "rb"))
        pca_score = pca_model.transform(flatten)[0]
    
    if (height < 150):
        pca_score = [1e4]*15 #arbitrary large number to remove this individual in the next gen
    
    #Target microstructure's score
    vcm = [ 0.283521796, -2.13598552, -1.41762233, 5.75928486, 1.71384479, -0.901034905, -0.586843370, 1.09576052, 
    0.00109143346, 0.229672674, 0.120327034, -0.711336532, 0.363371927, -0.190532327, -0.0478372960]
    lcm = [-0.56296494, 6.98643394, -3.946023, -0.04858549, -2.02767558, -3.35305158, 0.43746681, 0.28760718, 7.09752507, 
    -0.06149019, -1.14950567, -0.30897191, -0.65911122, -0.30734174, -0.05449982]
    hcm = [-0.453252864, -1.21187718, -0.478605694, 1.43706080e+00, 0.0161920288, 0.109714202, 0.226991095, 0.371122766, 
    -0.004239704145, 0.0235540240, 0.00176102675, -0.0498096860, -0.0196430132, 0.0758684162, -0.00121008798]
    
    target = lcm
    
    error_term = 0.0
    weight = [1.0]*15
    weight[0] = 10

    print ("PCA score = ", pca_score)
    print ("Target score = ", target)
    for i in range(len(target)):
        error_term += weight[i]*np.abs(pca_score[i] - target[i]) 

    #compute complexity of deposition rate
    input_params = np.array(params)

    a0_star = input_params[0]
    an_star = input_params[1:6]
    bn_star = input_params[6:11]
    #random_scaling_factor = input_params[11]
    P = int(input_params[11])

    it = np.arange(P)
    v_star_der =  - an_star[0]*np.sin(2*np.pi*it/P)*(2*np.pi/P) + bn_star[0]*np.cos(2*np.pi*it/P)*(2*np.pi/P) \
                  - an_star[1]*np.sin(2*np.pi*it*2/P)*(2*np.pi*2/P) + bn_star[1]*np.cos(2*np.pi*it*2/P)*(2*np.pi*2/P) \
                  - an_star[2]*np.sin(2*np.pi*it*3/P)*(2*np.pi*3/P) + bn_star[2]*np.cos(2*np.pi*it*3/P)*(2*np.pi*3/P) \
                  - an_star[3]*np.sin(2*np.pi*it*4/P)*(2*np.pi*4/P) + bn_star[3]*np.cos(2*np.pi*it*4/P)*(2*np.pi*4/P) \
                  - an_star[4]*np.sin(2*np.pi*it*5/P)*(2*np.pi*5/P) + bn_star[4]*np.cos(2*np.pi*it*5/P)*(2*np.pi*5/P)
    
    complexity_term = np.abs(v_star_der[1:] - v_star_der[0:-1]).sum()

    obj = error_term + complexity_term

    return obj



if __name__ == "__main__":
    inputfile = "in.test"
    params = [-1.0, 0.4053, 0.0, 0.045, 0.0, 0.0162, 0.0, 0.0, 0.0, 0.0, 0.0]
    write_input_file(inputfile, params)
    submit_job_array(0, inputfile, 1)
    obj = analyze_microstructure()

