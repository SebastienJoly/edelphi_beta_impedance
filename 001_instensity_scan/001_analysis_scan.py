import pickle

import numpy as np
from scipy.constants import c as clight
import scipy.io as sio

import mystyle as ms
import myfilemanager as mfm


scan_folder_rel = 'simulations'
intensity_scan = np.arange(0., 1.0001e11, 1e9)

# Plot settings
l_min_plot = -5
l_max_plot = 3
min_intensity_plot= 0
max_intensity_plot= 1e11
tau_min_plot = 0
tau_max_plot = 300
flag_tilt_lines = False
flag_mode0 = True

Omega_mat = []
M00_array = []
for ii, iint in enumerate(intensity_scan):

    fname = scan_folder_rel + (f'/intensity_{intensity_scan[ii]:.3e}'
            '/eigenvalues.mat')
    obcurr = mfm.myloadmat_to_obj(fname)

    Omega_mat.append(obcurr.Omega)
    M00_array.append(obcurr.M00_array)

Omega_mat = np.array(Omega_mat)
M00_array = np.array(M00_array)

omega0 = obcurr.omega0
omega_s = obcurr.omega_s
l_min = obcurr.l_min
l_max = obcurr.l_max
m_max = obcurr.m_max
N_max = obcurr.N_max

import matplotlib.pyplot as plt
plt.close('all')
ms.mystyle(fontsz=14, traditional_look=False)


fig500 = plt.figure(500)#, figsize=(1.3*6.4, 1.3*4.8))
ax = fig500.add_subplot(111)
ax.set_facecolor('grey')
im_min_col = 5
im_max_col = 200
im_min_size = 5
im_max_size = 50
import matplotlib
if flag_mode0:
    maskmode0 = np.abs(np.real(M00_array)/omega_s)<0.9
    ax.plot(intensity_scan[maskmode0], np.real(M00_array)[maskmode0]/omega_s, '--',
        linewidth=2, color='w', alpha=0.7)
for ii in range(len(intensity_scan)):
    Omega_ii = Omega_mat[ii, :]
    ind_sorted = np.argsort(-np.imag(Omega_ii))
    re_sorted = np.take(np.real(Omega_ii), ind_sorted)
    im_sorted = np.take(np.imag(Omega_ii), ind_sorted)
    plt.scatter(x=intensity_scan[ii]+0*np.imag(Omega_mat[ii, :]),
            y=re_sorted/omega_s,
            c = np.clip(-im_sorted, im_min_col, im_max_col),
            cmap=plt.cm.jet,
            s=np.clip(-im_sorted, im_min_size, im_max_size),
            vmin=im_min_col, vmax=im_max_col,
            norm=matplotlib.colors.LogNorm(),
            )
#plt.suptitle(title)
ax.set_xlim(min_intensity_plot, max_intensity_plot)
ax.set_ylim(l_min_plot, l_max_plot)
fig500.subplots_adjust(right=1.)
for lll in range(l_min_plot-10, l_max_plot+10):
    if flag_tilt_lines:
        add_to_line = np.array(DQ_0_list)*omega0/omega_s
    else:
        add_to_line = 0.
    ax.plot(intensity_scan,
            0*intensity_scan+ lll + add_to_line,
            color='w',
            alpha=.5, linestyle='--')
ax.tick_params(right=True, top=True)
plt.colorbar()

figtau = plt.figure(600)
axtau = figtau.add_subplot(111)
axtau.plot(intensity_scan, np.imag(Omega_mat),
        '.', color='grey', alpha=0.5)
axtau.plot(intensity_scan, np.max(-np.imag(Omega_mat), axis=1),
        linewidth=2, color='b')
axtau.set_xlim(min_intensity_plot, max_intensity_plot)
axtau.set_ylim(tau_min_plot, tau_max_plot)
axtau.grid(True, linestyle=':')
#figtau.suptitle(title)

plt.show()

