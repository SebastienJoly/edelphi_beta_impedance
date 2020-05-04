import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import m_p

import PyHEADTAIL.impedances.wakes as wakes
from PyHEADTAIL.particles import generators
from PyHEADTAIL.particles.slicing import UniformBinSlicer

##############
# Parameters #
##############

circumference = 27e3
gamma = 480.

# Impedance definition
resonator_R_shunt = 3*25e6
resonator_frequency = 2e9
resonator_Q = 1.
Yokoya_X1 = 1.
Yokoya_X2 = 1.
n_slices_wake = 500

# Bunch_parameters
intensity = 1.2e+11
sigma_z = 0.09705
z_cut = 2.5e-9/2*clight
particle_charge = qe
particle_mass = m_p

# e-delphi settings
n_sine_terms = 200
test_amplitude = 1.
detuning_fit_order = 10

###################
# Build impedance # 
###################

slicer_for_wakefields = UniformBinSlicer(
                        n_slices_wake, z_cuts=(-z_cut, z_cut))

wake_dipolar = wakes.Resonator(R_shunt=resonator_R_shunt,
        frequency=resonator_frequency,
        Q=resonator_Q,
        Yokoya_X1=Yokoya_X1,
        Yokoya_Y1=0.,
        Yokoya_X2=0.,
        Yokoya_Y2=0.,
        switch_Z=0)
wake_dipolar_element = wakes.WakeField(slicer_for_wakefields, wake_dipolar)

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

# Build response matrix for dipolar impedance

slicer_for_harmonicresponse = UniformBinSlicer(
                        n_sine_terms, z_cuts=(-z_cut, z_cut))

# Make a test bunch (I need only longitudinal profile and energy to be correct)
bunch = generators.ParticleGenerator(
    macroparticlenumber=n_sine_terms*1000,
    intensity=intensity,
    charge=particle_charge,
    mass=particle_mass,
    circumference=circumference,
    gamma=gamma,
    distribution_x=generators.gaussian2D(1e-9), # Dummy not really used
    alpha_x=0, beta_x=100., D_x=0.,
    distribution_y=generators.gaussian2D(1e-9),
    alpha_y=0., beta_y=100., D_y=0.,
    distribution_z=generators.cut_distribution(
        generators.gaussian2D_asymmetrical(sigma_u=sigma_z, sigma_up=1e-4),
        is_accepted=(lambda z, dp: np.array(len(z) * [True]))),
    ).generate()

# Generate configurations
assert(n_sine_terms % 2 ==0)
cos_ampl_list = []
sin_ampl_list = []
n_osc_list = []
for ii in range(n_sine_terms//2):
    cos_ampl_list.append(test_amplitude)
    sin_ampl_list.append(0.)
    n_osc_list.append(ii)

    cos_ampl_list.append(0.)
    sin_ampl_list.append(test_amplitude)
    n_osc_list.append(ii+1)

# cos_ampl_list = [100*1e-4]
# sin_ampl_list = [0]
# n_osc_list = [3]

# Measure responses
x_meas_mat = []
x_mat = []
dpx_mat = []
for itest in range(len(cos_ampl_list)):

    N_oscillations = n_osc_list[itest]
    sin_amplitude = sin_ampl_list[itest]
    cos_amplitude = cos_ampl_list[itest]

    # Recenter all slices
    slices_set = bunch.get_slices(slicer_for_harmonicresponse, statistics=True)
    for ii in range(slices_set.n_slices):
        ix = slices_set.particle_indices_of_slice(ii)
        if len(ix) > 0:
            bunch.x[ix] -= np.mean(bunch.x[ix])
            bunch.xp[ix] -= np.mean(bunch.xp[ix])

    # Get slice centers
    z_slices = slices_set.z_centers
    N_slices = len(z_slices)

    # Get z_step beween slices and define z_range
    z_step = z_slices[1] - z_slices[0]
    z_range = z_slices[-1] - z_slices[0] + z_step # Last term is to make
                                                  # sinusoids numerically
                                                  # orthogonal
    # Generate ideal sinusoidal distortion
    x_ideal = (sin_amplitude * np.sin(2*np.pi*N_oscillations*z_slices/z_range)
             + cos_amplitude * np.cos(2*np.pi*N_oscillations*z_slices/z_range))

    # Add sinusoidal distortion to particles
    bunch.x += sin_amplitude * np.sin(2*np.pi*N_oscillations*bunch.z/z_range)
    bunch.x += cos_amplitude * np.cos(2*np.pi*N_oscillations*bunch.z/z_range)

    # Measure
    bunch.clean_slices()
    slices_set = bunch.get_slices(slicer_for_harmonicresponse, statistics=True)
    x_slices = slices_set.mean_x
    int_slices = slices_set.lambda_bins()/qe
    bunch.clean_slices()

    # Apply impedance
    wake_dipolar_element.track(bunch)

    # Measure kicks
    bunch.clean_slices()
    slices_set = bunch.get_slices(slicer_for_harmonicresponse, statistics=True)
    dpx_slices = slices_set.mean_xp

    # Store results
    x_mat.append(x_ideal.copy())
    x_meas_mat.append(x_slices.copy())
    dpx_mat.append(dpx_slices.copy())

x_mat = np.array(x_mat)
x_meas_mat = np.array(x_meas_mat)
dpx_mat = np.array(dpx_mat)

import scipy.io as sio
sio.savemat('response_data.mat',{
    'x_mat': x_mat,
    'z_slices': z_slices,
    'dpx_mat': dpx_mat})


#############################
# Detuning characterization #
#############################

bunch.x *= 0
bunch.xp *= 0

bunch.x += test_amplitude
bunch.clean_slices()
wake_quadrupolar_element.track(bunch)
bunch.clean_slices()

slices_set = bunch.get_slices(slicer_for_harmonicresponse, statistics=True)
dpx_slices = slices_set.mean_xp

k_quad = slices_set.mean_xp/test_amplitude

p = np.polyfit(z_slices, k_quad, deg=detuning_fit_order)
alpha_N = p[::-1]

#############################
# Detuning characterization #
#############################
# Build matrix
MM_obj = CouplingMatrix(z_slices, HH, cloud_rescale_by*KK, l_min,
        l_max, m_max, n_phi, n_r, N_max, Q_full, sigma_b, r_b,
        a_param, lambda_param, omega0, omega_s, eta,
        alpha_p=alpha_N,
        beta_p = beta_N, beta_fun_rescale=beta_fun_rescale,
        include_detuning_with_longit_amplitude=include_detuning_with_long_amplitude,
        pool_size=pool_size)

