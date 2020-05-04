import os
import numpy as np

scan_folder_rel = 'simulations'
intensity_scan = np.arange(0., 1.0001e11, 1e9)

environment_preparation = f'''
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/venvs/py3/bin/activate
source /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/setpythopath
'''
# Last one is to get response matrix path

files_to_be_copied = [
        '../000_single_simulation/000_edelphi_impedance.py',
        '../000_single_simulation/mode_coupling_matrix.py',
        '../000_single_simulation/impedance_characterization.py',
        ]

settings_to_be_replaced_in = '000_edelphi_impedance.py'


scan_folder_abs = os.getcwd() + '/' + scan_folder_rel
os.mkdir(scan_folder_abs)

# prepare scan
for ii in range(len(intensity_scan)):

    # Make directory
    current_sim_ident= f'intensity_{intensity_scan[ii]:.3e}'
    print(current_sim_ident)
    current_sim_abs_path = scan_folder_abs+'/'+current_sim_ident
    os.mkdir(current_sim_abs_path)

    # Copy files
    for ff in files_to_be_copied:
        os.system(f'cp {ff} {current_sim_abs_path}')

    # Replace settings section
    settings_section = f'''# start-settings-section
this_intensity = {intensity_scan[ii]:.3e}
# end-settings-section'''


    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'r') as fid:
        lines = fid.readlines()
    istart = np.where([ss.startswith('# start-settings-section') for ss in lines])[0][0]
    iend = np.where([ss.startswith('# end-settings-section') for ss in lines])[0][0]
    with open(current_sim_abs_path+'/'+settings_to_be_replaced_in, 'w') as fid:
        fid.writelines(lines[:istart])
        fid.write(settings_section + '\n')
        fid.writelines(lines[iend+1:])

    # Prepare job script
    job_content = f'''#!/bin/bash

{environment_preparation}

# Environment preparation
echo PYTHONPATH=$PYTHONPATH
echo which python
which python

cd {current_sim_abs_path}

python 000_edelphi_impedance.py
'''
    with open(current_sim_abs_path + '/job.job', 'w') as fid:
       fid.write(job_content)

# Prepare htcondor cluster of jobs
import htcondor_config as htcc
htcc.htcondor_config(
        scan_folder_abs,
        time_requirement_days=0.1,
        htcondor_files_in=scan_folder_abs)
