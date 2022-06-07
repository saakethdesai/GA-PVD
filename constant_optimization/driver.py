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


def write_input_file(inputfile, params):

    header = ("! *** Phase-field model settings\n"
              "phase-field_model                 alloy_deposition\n")

    #a = str(params[0]) + "d0"
   
    film_params = ("initial_film-height               10.0d0\n"
                   "surface_roughness                 0.0d0 0.00d0 0.0d0\n"
                   "composition_distribution          0.0d0 0.35d0 -1.0d0 1.0d0\n"
                   "growth-restriction_value          0.75d0\n"
                   "growth-limit_value                0.40d0\n")
    
    b = str(params[0]) + "d0"
    c = str(params[1]) + "d0"
    d = str(params[2]) + "d0"
    
    free_energy_params = ("mobility_coefficient_phi_function constant\n"
                          "mobility_coefficient_phi_value    %s\n"
                          "free-energy_barrier_phi           1.0d0\n"
                          "gradient-energy_coefficient_phi   1.0d0\n"
                          "surface-localization_chi          0.2d0\n"
                          "surface_mobilities_chi_function   constant\n"
                          "surface_mobilities_chi_value      %s %s\n"
                          "mobility_coefficients_chi_function constant\n"
                          "mobility_coefficients_chi_value   %s %s\n"
                          "free-energy_barrier_chi           1.0d0\n"
                          "gradient-energy_coefficient_chi   1.0d0\n") %(b, c, c, d, d)

    e = str(params[3]) + "d0"
    f = str(params[4]) + "d0"
    
    deposition_params  = ("vapor_diffusivity_function        constant\n"
                          "vapor_diffusivity_value           %s\n"
                          "initial_mass-density              1.0d0\n"
                          "velocity_function                 constant\n"
                          "velocity-field_components         0.0d0 -%s 0.0d0\n") %(e, f)

    sim_time = int(np.abs(0.5*512/(params[4]*1e-3) * 7)) 

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
    
    account = "fy220020"
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
        pca_model = pkl.load(open("/qscratch/saadesa/ml_thin_films/pca_training_set_constant_runs/train_models/08/model.pkl", "rb"))
        pca_score = pca_model.transform(flatten)[0]
    
    if (height < 150):
        pca_score = [1e4]*15 #arbitrary large number to remove this individual in the next gen
    
    #Target microstructure's score
    vcm = [8.52535223, -2.77617664, 1.23666904, -0.43520697, 7.43297295, 0.69987426, 1.7374482, -0.14193999]
    lcm = [7.72619294, 6.72236927, 3.37650827, -7.33597607, -0.3806111, -3.6652979, 1.90657407, 0.61086835]
    npcm = [-42.5664692, -0.650754612, 0.142409535, -0.0310442473, -0.182995374, -0.249923881, -0.190529707, 0.0842723311]
    rcm = [-6.19554763, -1.18575743, 0.23018625, -0.05399898, -0.38560533, -0.12611856, -0.25905995, 0.0325774] 
    
    target = rcm
    
    error_term = 0.0
    weight = [1.0]*8
    weight[0] = 10

    print ("PCA score = ", pca_score)
    print ("Target score = ", target)
    for i in range(len(target)):
        error_term += weight[i]*np.abs(pca_score[i] - target[i]) 

    obj = error_term 

    return obj



if __name__ == "__main__":
    inputfile = "in.test"
    params = [-1.0, 0.4053, 0.0, 0.045, 0.0, 0.0162, 0.0, 0.0, 0.0, 0.0, 0.0]
    write_input_file(inputfile, params)
    submit_job_array(0, inputfile, 1)
    obj = analyze_microstructure()

