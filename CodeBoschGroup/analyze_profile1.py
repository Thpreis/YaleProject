import sys
import getopt

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from satellite_kinematics import utils

try:
    opts, args = getopt.getopt(
        sys.argv[1:], 'p1:p2:p3:s1:l1:s2:l2:o:n:',
        ['profile_1=', 'profile_2=', 'profile_3=', 'sdss_1=', 'label_1=',
         'sdss_2=', 'label_2=', 'output=', 'n_mocks=', 'names', 'abc_sdss'])
except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

PROFILES = [None, None, None]
N_MOCKS = 0
SDSS_1 = None
SDSS_2 = None
OUTPUT = None
LABEL_1 = 'SDSS'
LABEL_2 = 'SDSS'
NAMES = False
ABC_SDSS = False

for opt, arg in opts:
    if opt in ('-p1', '--profile_1'):
        PROFILES[0] = arg
    if opt in ('-p2', '--profile_2'):
        PROFILES[1] = arg
    if opt in ('-p3', '--profile_3'):
        PROFILES[2] = arg
    if opt in ('-s1', '--sdss_1'):
        SDSS_1 = arg
    if opt in ('-l1', '--label_1'):
        LABEL_1 = arg
    if opt in ('-s2', '--sdss_2'):
        SDSS_2 = arg
    if opt in ('-l2', '--label_2'):
        LABEL_2 = arg
    if opt in ('-o', '--output'):
        OUTPUT = arg
    if opt in ('-n', '--n_mocks'):
        N_MOCKS = int(arg)
    if opt in ('--names'):
        NAMES = True
    if opt in ('--abc_sdss'):
        ABC_SDSS = True

if ABC_SDSS:
    utils.a_r = 2.210
    utils.b_r = 0.478
    utils.c_r = 0.275
    utils.a_b = 2.142
    utils.b_b = 0.402
    utils.c_b = -0.170

utils.x_r_s = 0.5

print("Analyzing radial profile...")
if SDSS_1 is not None:
    print("First SDSS catalogue: %s" % SDSS_1)
if SDSS_2 is not None:
    print("Second SDSS catalogue: %s" % SDSS_2)
for i in range(3):
    if PROFILES[i] is not None:
        print("Profile %d mocks: %s" % (i + 1, PROFILES[i]))
print("Number of mocks: %d" % N_MOCKS)
print("Output: %s.*" % OUTPUT)

rp_bins = np.logspace(-2, np.log10(2), 10)
log_lum_bins = np.linspace(9.75, 11, 6)

if not NAMES:
    labels = [r'$\mathcal{R} = 1$, $\gamma = 1$',
              r'$\mathcal{R} = 2$, $\gamma = 1$',
              r'$\mathcal{R} = 2$, $\gamma = 0$']
else:
    labels = ['NFW', 'bNFW', 'Cored']

cm = matplotlib.cm.get_cmap('plasma')

plt.figure(figsize=(7.0, 4.5))
ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 1), colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=2)
ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
axarr = [ax1, ax2, ax3, ax4, ax5]

for i in range(3):
    if PROFILES[i] is not None:

        f = h5py.File(PROFILES[i] + 'parameters.hdf5', "r")
        assert f.attrs['profile'] == (i+1)

        sigma_mocks = []

        for k in range(N_MOCKS):

            catalog = utils.read_catalog_from_file(
                    PROFILES[i] + 'mock_%d.hdf5' % k)
            sigma_mocks.append(utils.surface_density_from_catalog(
                    catalog, log_lum_bins, rp_bins))

        sigma_mocks = np.array(sigma_mocks)

        for k in range(len(log_lum_bins) - 1):

            rp_med = np.sqrt(rp_bins[1:] * rp_bins[:-1])

            mask = ((rp_bins[1:] < utils.cylinder_size(
                     np.atleast_1d(10**log_lum_bins[k]),
                     np.atleast_1d('red'))[2][0]) |
                    (rp_bins[1:] < utils.cylinder_size(
                     np.atleast_1d(10**log_lum_bins[k]),
                     np.atleast_1d('blue'))[2][0]))

            axarr[k].fill_between(
                    np.log10(rp_med)[mask],
                    np.nanpercentile(rp_med * sigma_mocks[:, k, :], 16,
                                     axis=0)[mask],
                    np.nanpercentile(rp_med * sigma_mocks[:, k, :], 84,
                                     axis=0)[mask],
                    alpha=0.5, zorder=-99, label=labels[i],
                    facecolor=cm(0.6 * (i / 2.0) + 0.2), lw=0.5,
                    edgecolor='black')

for i in range(2):
    if i == 0:
        SDSS = SDSS_1
        LABEL = LABEL_1
        color = 'royalblue'
    else:
        SDSS = SDSS_2
        LABEL = LABEL_2
        color = 'tomato'
    if SDSS is not None:
        catalog = utils.read_catalog_from_file(SDSS)
        sigma_sdss = utils.surface_density_from_catalog(
            catalog, log_lum_bins, rp_bins)

        for k in range(len(log_lum_bins) - 1):

            mask = ((rp_bins[1:] < utils.cylinder_size(
                     np.atleast_1d(10**log_lum_bins[k]),
                     np.atleast_1d('red'))[2][0]) |
                    (rp_bins[1:] < utils.cylinder_size(
                     np.atleast_1d(10**log_lum_bins[k]),
                     np.atleast_1d('blue'))[2][0]))
            axarr[k].scatter(
                np.log10(rp_med)[mask], (rp_med * sigma_sdss[k])[mask],
                label=LABEL, color=color, marker='x')

for k in range(len(log_lum_bins) - 1):

    ymin, ymax = axarr[k].get_ylim()
    axarr[k].set_ylim(0, ymax * 1.3)
    axarr[k].set_xlabel(r'$\log r_p [h^{-1} \ \mathrm{Mpc}]$')

    axarr[k].set_xlim(np.amin(np.log10(rp_med)), np.amax(np.log10(rp_med)))
    if k == 0 or k == 2:
        axarr[k].set_ylabel(
            r'$r_p \times \Sigma(r_p) [h \ \mathrm{Mpc}^{-1}]$')

    if k == 1:
        axarr[k].legend(loc='upper right', bbox_to_anchor=(1.5, 0.75))

    axarr[k].annotate(
        r'$\log L_{\rm pri} = %.2f - %.2f$' %
        (log_lum_bins[k], log_lum_bins[k + 1]),
        xy=(.05, .95), xycoords='axes fraction', horizontalalignment='left',
        verticalalignment='top')

plt.tight_layout(pad=0.3)
plt.savefig(OUTPUT + '.pdf')
plt.savefig(OUTPUT + '.png', dpi=300)
