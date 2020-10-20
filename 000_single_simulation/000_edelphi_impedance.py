# Parameters from config script

# start-settings-section

this_intensity = 1.2e11

# end-settings-section

##########################################
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import m_p

import PyHEADTAIL.impedances.wakes as wakes
from PyHEADTAIL.particles.slicing import UniformBinSlicer

from mode_coupling_matrix import CouplingMatrix
import impedance_characterization as ic

##############
# Parameters #
##############

# Choose plane, either 'x' or 'y'
plane = ['x']

circumference = 27e3
particle_gamma = 480.
beta_fun_at_imped = 92.7
Q_full = 62.27
Qp= 0.
Qs = 4.9e-3
eta = 0.000318152589

omega0 = 2*np.pi*clight/circumference
omega_s = Qs * omega0

# Impedance definition
resonator_R_shunt = 10e6
resonator_frequency = 1e9
resonator_Q = 1.
Yokoya_X1 = 1.
Yokoya_X2 = 1.
n_slices_wake = 500

# Bunch_parameters
intensity = this_intensity
sigma_z = 0.09705
z_cut = 2.5e-9/2*clight
particle_charge = qe
particle_mass = m_p

# e-delphi settings
n_samples_hh_kk = 200
test_amplitude = 1.
detuning_fit_order = 10
l_min = -7
l_max = 7
m_max = 20
n_phi = 3*360
n_r = 3*200
N_max = 29
n_tail_cut = 0
include_detuning_with_long_amplitude = True
r_b = 4*sigma_z
a_param = 8./r_b**2
lambda_param = 1

pool_size = 0 # N cores (0 for serial)

###################
# Build impedance # 
###################

# Slicer
slicer_for_wakefields = UniformBinSlicer(
                        n_slices_wake, z_cuts=(-z_cut, z_cut))

# Dipolar wake
wake_dipolar = wakes.Resonator(R_shunt=resonator_R_shunt,
        frequency=resonator_frequency,
        Q=resonator_Q,
        Yokoya_X1=Yokoya_X1,
        Yokoya_Y1=0.,
        Yokoya_X2=0.,
        Yokoya_Y2=0.,
        switch_Z=0)
wake_dipolar_element = wakes.WakeField(slicer_for_wakefields, wake_dipolar)

# Quadrupolar wake
wake_quadrupolar = wakes.Resonator(R_shunt=resonator_R_shunt,
        frequency=resonator_frequency,
        Q=resonator_Q,
        Yokoya_X1=0.,
        Yokoya_Y1=0.,
        Yokoya_X2=Yokoya_X2,
        Yokoya_Y2=0.,
        switch_Z=0)
wake_quadrupolar_element = wakes.WakeField(slicer_for_wakefields,
        wake_quadrupolar)

##########################
# Characterize impedance # 
##########################
print('Start impedance characterization...')
imp_characterization = ic.characterize_impedances(
        wake_dipolar_element=wake_dipolar_element,
        wake_quadrupolar_element=wake_quadrupolar_element,
        n_samples_hh_kk=n_samples_hh_kk,
        test_amplitude=test_amplitude,
        intensity=intensity,
        sigma_z=sigma_z,
        circumference=circumference,
        particle_charge=particle_charge,
        particle_mass=particle_mass,
        particle_gamma=particle_gamma,
        z_cut=z_cut,
        n_tail_cut=n_tail_cut,
        detuning_fit_order=detuning_fit_order)
print('Done!')


##############################
# Build mode coupling matrix #
##############################
# Build matrix
assert(N_max < n_samples_hh_kk/4)
# Raise error if plane is different from 'x' or 'y'
if (plane != 'x') and (plane != 'y'):
    raise ValueError('Wrong argument for plane')
beta_N = [0, Qp]

MM_obj = CouplingMatrix(
            imp_characterization['z_slices'],
            imp_characterization['HH_' + plane],
            imp_characterization['KK_' + plane],
            l_min, l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_z, r_b,
            a_param, lambda_param, omega0, omega_s, eta,
            alpha_p=imp_characterization['alpha_N' + plane],
            beta_p = beta_N, beta_fun_rescale=beta_fun_at_imped,
            include_detuning_with_longit_amplitude=include_detuning_with_long_amplitude,
            pool_size=pool_size)

print('Matrix built !')


#######################
# Compute eigenvalues #
#######################
Omega = MM_obj.compute_mode_complex_freq(omega_s)

i_l0 = np.argmin(np.abs(MM_obj.l_vect))
M00_array = MM_obj.MM[i_l0,0,i_l0,0]

print('Eigenvalues computed !')


import scipy.io as sio
sio.savemat('eigenvalues.mat', {
    'Omega': Omega,
    'M00_array': M00_array,
    'plane': plane,
    'omega0': omega0,
    'omega_s': omega_s,
    'l_min': l_min,
    'l_max': l_max,
    'm_max': m_max,
    'N_max': N_max})

print('Finished')
