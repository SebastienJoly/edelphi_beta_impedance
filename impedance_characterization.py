import numpy as np

from scipy.constants import e as qe

from PyHEADTAIL.particles import generators
from PyHEADTAIL.particles.slicing import UniformBinSlicer

def characterize_impedances(wake_dipolar_element, wake_quadrupolar_element,
        n_samples_hh_kk, test_amplitude,
        intensity, sigma_z,
        circumference, particle_charge, particle_mass, particle_gamma,
        z_cut, n_tail_cut, detuning_fit_order):


    # Build response matrix for dipolar impedance
    slicer_for_harmonicresponse = UniformBinSlicer(
                            n_samples_hh_kk, z_cuts=(-z_cut, z_cut))

    # Make a test bunch (I need only longitudinal profile and energy to be correct)
    bunch = generators.ParticleGenerator(
        macroparticlenumber=n_samples_hh_kk*1000,
        intensity=intensity,
        charge=particle_charge,
        mass=particle_mass,
        circumference=circumference,
        gamma=particle_gamma,
        distribution_x=generators.gaussian2D(1e-9), # Dummy not really used
        alpha_x=0, beta_x=100, D_x=0.,
        distribution_y=generators.gaussian2D(1e-9),
        alpha_y=0., beta_y=100., D_y=0.,
        distribution_z=generators.cut_distribution(
            generators.gaussian2D_asymmetrical(sigma_u=sigma_z, sigma_up=1e-4),
            is_accepted=(lambda z, dp: np.array(len(z) * [True]))),
        ).generate()

    bunch.x *= 0
    bunch.xp *= 0

    # Generate configurations
    assert(n_samples_hh_kk % 2 ==0)
    cos_ampl_list = []
    sin_ampl_list = []
    n_osc_list = []
    for ii in range(n_samples_hh_kk//2):
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

    HH = x_mat
    KK = dpx_mat

    if n_tail_cut > 0:
        KK[:, :n_tail_cut] = 0.
        KK[:, -n_tail_cut:] = 0.

    #############################
    # Detuning characterization #
    #############################
    if wake_quadrupolar_element is not None:
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

    wake_characterization = {
            'HH': HH,
            'KK': KK,
            'z_slices': z_slices,
            'alpha_N': alpha_N}

    return wake_characterization

