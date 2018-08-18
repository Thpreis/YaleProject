import os
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import h5py

from halotools.sim_manager import CachedHaloCatalog
from satellite_kinematics import utils, AnalyticModel

from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

DIRECTORY = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:', ['directory='])
except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

for opt, arg in opts:
    if opt in ('-d', '--directory'):
        DIRECTORY = arg


i_max = 0
while os.path.isdir(DIRECTORY + 'fit_%d_minimize' % (i_max + 1)):
    i_max += 1

parameters = h5py.File(
    DIRECTORY + 'fit_%d_minimize/parameters.hdf5' % i_max, 'r').attrs

lf_sys_err = parameters['lf_sys_err']
covariance = parameters['covariance']
bias = parameters['bias']
halocat = parameters['halocat']
profile = parameters['profile']
catalog = parameters['catalog']

utils.a_r = 2.210
utils.b_r = 0.478
utils.c_r = 0.275
utils.a_b = 2.142
utils.b_b = 0.402
utils.c_b = -0.170

# %% Let's import the covariance matrix, the bias and which data points to use.

f = h5py.File(covariance + 'constraints.hdf5', 'r')

cov = f['data_cov'][:]
use = f['data_use'][:]

n_s = f.attrs['n_mocks']

f.close()

f = h5py.File(bias + 'constraints.hdf5', 'r')

use = use & f['data_use'][:]

bias = f['data_bias'][:]
bias[:10] = 0.0

f.close()

# %% Next, we analyze the mock we want to fit to.

catalog = utils.read_catalog_from_file(catalog)
theta_obs = utils.constraints_dict_to_vector(
    utils.constraints_dict_from_spec_catalog(catalog, profile=profile))
use = use & ((theta_obs != 0) & ~np.isnan(theta_obs))

# %% Then, we load the analytic model.

if halocat == 'smdpl':
    halocat = CachedHaloCatalog(simname='smdpl', redshift=utils.z_halo,
                                dz_tol=0.01)
elif halocat == 'conplanck':
    halocat = CachedHaloCatalog(simname='conplanck', redshift=utils.z_halo,
                                version_name='1', dz_tol=0.01)

model = AnalyticModel(halocat, profile=profile, z_halo=utils.z_halo,
                      a_r=utils.a_r, b_r=utils.b_r, c_r=utils.c_r,
                      a_b=utils.a_b, b_b=utils.b_b, c_b=utils.c_b,
                      threshold=np.amin(utils.default_log_lum_bins))

# %% Here, we calculate the precision matrix.

for i, row in enumerate(cov):
    if i < 10:
        cov[i, i] = cov[i, i] + theta_obs[i]**2 * lf_sys_err**2

theta_obs = theta_obs[use]

cov_use = []
for i, row in enumerate(cov):
    if use[i]:
        cov_use.append(row[use])
cov_use = np.array(cov_use)

n_d = np.sum(use)
pre = (float(n_s - n_d - 2) / float(n_s - 1)) * np.linalg.inv(cov_use)

# %%
n_d = 16 # number of free parameters
chi = np.linspace(0, 100, 1e7)
pdf = chi**(n_d - 1) * np.exp(- chi**2 / 2.0)
cdf = cumtrapz(pdf, x=chi)
cdf = cdf / cdf[-1]

cdf_inv = interp1d(cdf, chi[1:]**2)
print("chi squared for 68%%: %.2f" % cdf_inv(0.6827))

chi_squared = np.zeros(i_max + 1)

for i in range(len(chi_squared)):
    if i > 0:
        model.param_dict.update(utils.get_best_fit_param_dict(
            DIRECTORY + 'fit_%d_minimize/' % i))

    theta_mod = utils.constraints_dict_to_vector(model.get_constraints_dict())
    theta_mod = (theta_mod + bias)[use]
    theta_diff = theta_mod - theta_obs

    chi_squared[i] = np.inner(np.inner(theta_diff, pre), theta_diff)
    print("Iteration %d: %.1f" % (i, chi_squared[i]))

plt.plot(np.log10(chi_squared - chi_squared[-1] + 1), marker='o',
         color='black')
plt.axhline(np.log10(1 + cdf_inv(0.6827)), alpha=0.5, color='black', ls='--')
plt.xticks([0, 1, 2, 3, 4, 5],
           [r"$\boldsymbol{\theta}_0$", r"$\boldsymbol{\theta}_1$",
            r"$\boldsymbol{\theta}_2$", r"$\boldsymbol{\theta}_3$",
            r"$\boldsymbol{\theta}_4$", r"$\boldsymbol{\theta}_5$"])
plt.minorticks_off()
plt.xlim(-0.5, i_max + 0.5)
plt.ylim(bottom=-0.1)
plt.xlabel(r'Model')
plt.ylabel(r'$\log (\Delta \chi^2 + 1)$')
plt.tight_layout(pad=0.3)
plt.savefig(DIRECTORY + 'progress.pdf')
