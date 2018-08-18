import sys
import getopt
import os
from mpi4py import MPI

import h5py
import numpy as np
import pymultinest
from halotools.sim_manager import CachedHaloCatalog
from astropy.table import Table

from satellite_kinematics import utils, AnalyticModel

COVARIANCE = None
BIAS = None
PROFILE = 0
LF_SYS_ERR = 0.02
HALOCAT = 'smdpl'
OUTPUT = None
MODE = 'normal'
SDSS = False
NO_KINEMATICS = False
GAMMA_3 = False
BETA_F = False
ZETA = False
D_SIGMA = False
CONVERGENCE = False

try:
    opts, args = getopt.getopt(
        sys.argv[1:], 'i:c:b:p:e:h:o:m:sk',
        ['input=', 'covariance=', 'bias=', 'profile=', 'err_lf=', 'halocat=',
         'output=', 'mode=', 'sdss', 'no_kinematics', 'gamma_3', 'beta_f',
         'zeta', 'd_sigma'])
except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

for opt, arg in opts:
    if opt in ('-c', '--covariance'):
        COVARIANCE = arg
    elif opt in ('-b', '--bias'):
        BIAS = arg
    elif opt in ('-p', '--profile'):
        PROFILE = int(arg)
    elif opt in ('-e', '--err_lf'):
        LF_SYS_ERR = arg
    elif opt in ('-h', '--halocat'):
        HALOCAT = arg
    elif opt in ('-o', '--output'):
        OUTPUT = arg
    elif opt in ('-m', '--mode'):
        MODE = arg
    elif opt in ('-s', '--sdss'):
        SDSS = True
    elif opt in ('-k', '--no_kinematics'):
        NO_KINEMATICS = True
    elif opt in ('--gamma_3'):
        GAMMA_3 = True
    elif opt in ('--beta_f'):
        BETA_F = True
    elif opt in ('--zeta'):
        ZETA = True
    elif opt in ('--d_sigma'):
        D_SIGMA = True

FITTING = Table([[], [], [], [], []],
                names=['name', 'fit', 'min', 'max', 'default'],
                dtype=['S20', np.bool, np.float32, np.float32, np.float32])

FITTING.add_row(['log_L_0_r', True, 9.0, 10.5, 9.99])
FITTING.add_row(['log_M_1_r', True, 10.0, 13.0, 11.50])
FITTING.add_row(['gamma_1_r', True, 0.0, 5.0, 4.88])
FITTING.add_row(['gamma_2_r', True, 0.0, 2.0, 0.31])
FITTING.add_row(['gamma_3_r', False, -0.3, +0.3, 0.00])
FITTING.add_row(['sigma_r', True, 0.10, 0.25, 0.20])
FITTING.add_row(['log_L_0_b', True, 9.0, 10.5, 9.55])
FITTING.add_row(['log_M_1_b', True, 10.0, 13.0, 10.55])
FITTING.add_row(['gamma_1_b', True, 0.0, 5.0, 2.13])
FITTING.add_row(['gamma_2_b', True, 0.0, 2.0, 0.34])
FITTING.add_row(['gamma_3_b', False, -0.3, +0.3, 0.00])
FITTING.add_row(['sigma_b', True, 0.10, 0.4, 0.24])
FITTING.add_row(['f_0', True, 0.0, 1.0, 0.70])
FITTING.add_row(['alpha_f', True, -0.5, 1.0, 0.15])
FITTING.add_row(['beta_f', False, -0.5, 0.5, 0.00])
FITTING.add_row(['a_1', True, 0.5, 1.1, 0.82])
FITTING.add_row(['a_2', False, 0, 3.0, 0.0])
FITTING.add_row(['log_M_2', False, 12.0, 15.0, 14.28])
FITTING.add_row(['b_0', True, -1.5, 0.5, -0.766])
FITTING.add_row(['b_1', True, 0.0, 2.0, 1.008])
FITTING.add_row(['b_2', True, -0.5, 0.5, -0.094])
FITTING.add_row(['zeta', False, 0.5, 2.0, 1.0])
FITTING.add_row(['d_sigma_r', False, -0.1, 0.1, 0.0])
FITTING.add_row(['d_sigma_b', False, -0.1, 0.1, 0.0])

if SDSS:
    utils.a_r = 2.210
    utils.b_r = 0.478
    utils.c_r = 0.275
    utils.a_b = 2.142
    utils.b_b = 0.402
    utils.c_b = -0.170

for i, name in enumerate(FITTING['name']):
    if 'gamma_3' in name and GAMMA_3:
        FITTING['fit'][i] = True
    if 'beta_f' in name and BETA_F:
        FITTING['fit'][i] = True
    if 'zeta' in name and ZETA:
        FITTING['fit'][i] = True
    if 'd_sigma' in name and D_SIGMA:
        FITTING['fit'][i] = True

mask = FITTING['fit']
SEL = np.cumsum(mask) - 1
N_PARAMS = np.sum(mask)
THETA_MIN = FITTING['min'][mask]
THETA_MAX = FITTING['max'][mask]

EVIDENCE_TOL = 0.0001
SAMPLING_EFF = 0.005
N_LIVE = 10000
CONST_EFF = False

if MODE == 'convergence':
    N_LIVE = 20000
    SAMPLING_EFF = 0.002
elif MODE == 'minimize':
    SAMPLING_EFF = 0.5
    CONST_EFF = True
elif MODE == 'normal':
    pass
else:
    print("Unkown mode of operation! Got '%s', expected 'normal', " +
          "'convergence' or 'minimze'.")
    sys.exit(2)

# %% Let's write the fitting parameters into an extra file.
if not os.path.exists(OUTPUT) and MPI.COMM_WORLD.Get_rank() == 0:
    os.makedirs(OUTPUT)
    FITTING.write(OUTPUT + 'parameters.hdf5', path='parameters',
                  overwrite=True)
    f = h5py.File(OUTPUT + 'parameters.hdf5', "a")
    f.attrs['catalog'] = 'sdss/mock_7.hdf5'
    f.attrs['lf_sys_err'] = LF_SYS_ERR
    f.attrs['covariance'] = COVARIANCE
    f.attrs['bias'] = BIAS
    f.attrs['halocat'] = HALOCAT
    f.attrs['profile'] = PROFILE

# %% We first import the covariance matrix, bias and what data values to use.
f = h5py.File(COVARIANCE + 'constraints.hdf5', 'r')

cov = f['data_cov'][:]
use = f['data_use'][:]

n_s = f.attrs['n_mocks']

f.close()

f = h5py.File(BIAS + 'constraints.hdf5', 'r')

use = use & f['data_use'][:]

bias = f['data_bias'][:]
bias[:10] = 0.0

if NO_KINEMATICS:
    use[40:80] = False

f.close()

# %% Next, we analyze the mock we want to fit to.

theta_obs_all = []

for i in range(100):
    catalog = utils.read_catalog_from_file('sdss/mock_%d.hdf5' % (i + 6))
    theta_obs = utils.constraints_dict_to_vector(
        utils.constraints_dict_from_spec_catalog(catalog, profile=PROFILE))
    use = use & ((theta_obs != 0) & ~np.isnan(theta_obs))
    theta_obs_all.append(theta_obs)

# %% Then, we load the analytic model.

if HALOCAT == 'smdpl':
    halocat = CachedHaloCatalog(simname='smdpl', redshift=utils.z_halo,
                                dz_tol=0.01)
elif HALOCAT == 'conplanck':
    halocat = CachedHaloCatalog(simname='conplanck', redshift=utils.z_halo,
                                dz_tol=0.01, version_name='1')
model = AnalyticModel(halocat, threshold=9.5, profile=PROFILE,
                      a_r=utils.a_r, b_r=utils.b_r, c_r=utils.c_r,
                      a_b=utils.a_b, b_b=utils.b_b, c_b=utils.c_b)

# %% Here, we calculate the precision matrix.

for i, row in enumerate(cov):
    if i < 10:
        cov[i, i] = cov[i, i] + theta_obs[i]**2 * LF_SYS_ERR**2

cov_use = []
for i, row in enumerate(cov):
    if use[i]:
        cov_use.append(row[use])
cov_use = np.array(cov_use)

n_d = np.sum(use)
pre = (float(n_s - n_d - 2) / float(n_s - 1)) * np.linalg.inv(cov_use)

# %% Now we are ready to fit.


def prior(cube, n_dim, n_params):
    for i in range(n_dim):
        cube[i] = cube[i] * (THETA_MAX[i] - THETA_MIN[i]) + THETA_MIN[i]


def loglike(cube, ndim, nparams, lnew):

    for i in range(len(FITTING)):

        name = FITTING['name'][i]

        if FITTING['fit'][i]:
            model.param_dict[name] = cube[SEL[i]]
        else:
            model.param_dict[name] = FITTING['default'][i]

    theta_mod = utils.constraints_dict_to_vector(model.get_constraints_dict())
    theta_mod = (theta_mod + bias)[use]

    llik = 0

    for i in range(100):
        theta_diff = theta_mod - theta_obs_all[i][use]
        llik -= np.inner(np.inner(theta_diff, pre), theta_diff) / 2.0

    return llik

pymultinest.run(loglike, prior, N_PARAMS, resume=True, verbose=False,
                evidence_tolerance=EVIDENCE_TOL, n_live_points=N_LIVE,
                sampling_efficiency=SAMPLING_EFF, outputfiles_basename=OUTPUT,
                const_efficiency_mode=CONST_EFF, n_iter_before_update=1000,
                importance_nested_sampling=False)
