import sys
import getopt
import os
from mpi4py import MPI

import h5py
import numpy as np
import pymangle
from astropy.io import fits
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import TrivialPhaseSpace, SubhaloPhaseSpace
from halotools.empirical_models import HodModelFactory

from satellite_kinematics import utils, Lange18Cens, Lange18Sats

N_MOCKS = 0
I_JOB = MPI.COMM_WORLD.Get_rank()
N_JOB = MPI.COMM_WORLD.Get_size()
OUTPUT = None
FIT = None
HALOCAT = 'smdpl'
PROFILE = 0
TRIM = True
POSTERIOR = False
HIGH_Z = False

try:
    opts, args = getopt.getopt(
        sys.argv[1:], 'm:o:f:h:p:t:',
        ['mocks=', 'output=', 'fit=', 'halocat=', 'profile=', 'trim=',
         'posterior', 'high_z'])
except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

for opt, arg in opts:
    if opt in ('-m', '--mocks'):
        N_MOCKS = int(arg)
    elif opt in ('-o', '--output'):
        OUTPUT = arg
    elif opt in ('-f', '--fit'):
        if arg is None:
            FIT = None
        else:
            FIT = arg
    elif opt in ('-h', '--halocat'):
        HALOCAT = arg
    elif opt in ('-p', '--profile'):
        try:
            PROFILE = int(arg)
        except ValueError:
            PROFILE = arg
    elif opt in ('-t', '--trim'):
        if "False" in arg:
            TRIM = False
        elif "True" in arg:
            TRIM = True
        else:
            print("Argument trim must be True or False!")
            sys.exit(2)
    if opt in ('--posterior'):
        POSTERIOR = True
    if opt in ('high_z'):
        HIGH_Z = True

if HIGH_Z:
    utils.z_min = 0.067
    utils.z_max = 0.080
    utils.log_lum_min = 9.65

if I_JOB == 0:
    print("Building mock catalogs...")
    print("Number of mocks: %d" % N_MOCKS)
    print("Halo catalog: %s" % HALOCAT)
    print("Radial profile: %s" % PROFILE)
    print("Fit parameters: %s" % FIT)
    print("Output: %s" % OUTPUT)
    print("Draw models from posterior: %s" % POSTERIOR)

if POSTERIOR and FIT is None:
    print("No directory for fit given! Cannot draw from posterior!")
    sys.exit(2)

use = np.zeros(N_MOCKS, dtype=np.bool)
use[I_JOB::N_JOB] = True

halocat = CachedHaloCatalog(simname=HALOCAT, redshift=utils.z_halo,
                            dz_tol=0.01)

if PROFILE in [1, 2, 3]:
    model = utils.default_model(threshold=9.0, profile=PROFILE)
elif PROFILE == 'subhalo':
    cens_occ_model = Lange18Cens(threshold=9.0, redshift=utils.z_halo)
    cens_occ_model._suppress_repeated_param_warning = True
    cens_prof_model = TrivialPhaseSpace(redshift=utils.z_halo)
    sats_occ_model = Lange18Sats(threshold=9.0, redshift=utils.z_halo)

    m_vir_max = np.max(halocat.halo_table['halo_mvir'])
    host_haloprop_bins = np.logspace(11.5, np.log10(m_vir_max) + 0.1, 25)
    host_haloprop_bins[0] = 0
    sats_prof_model = SubhaloPhaseSpace(
        'satellites', redshift=utils.z_halo,
        host_haloprop_bins=host_haloprop_bins)

    model = HodModelFactory(centrals_occupation=cens_occ_model,
                            centrals_profile=cens_prof_model,
                            satellites_occupation=sats_occ_model,
                            satellites_profile=sats_prof_model)
else:
    print("Unkown radial profile for subhalos!")
    sys.exit(2)

if FIT is not None:
    model.param_dict = utils.get_best_fit_param_dict(FIT)
    if POSTERIOR:
        param_dict_sample = np.random.choice(
            utils.get_param_dict_sample(FIT), size=N_MOCKS)

if not os.path.exists(OUTPUT) and I_JOB == 0:
    os.makedirs(OUTPUT)
    f = h5py.File(OUTPUT + 'parameters.hdf5', "a")
    f.attrs['profile'] = PROFILE
    f.attrs['halocat'] = HALOCAT
    f.attrs['n_mocks'] = N_MOCKS
    f.attrs.update(model.param_dict)
    f.close()

window = pymangle.Mangle(
    utils.root_directory + "/sdss/window_pix.dr72bright0.ply")
survey_mask = pymangle.Mangle(
    utils.root_directory + "/sdss/mask_pix.dr72bright0.ply")
combmask = pymangle.Mangle(
    utils.root_directory + "/sdss/lss_combmask_pix.dr72.ply")
combmask_info = fits.getdata(
    utils.root_directory + "/sdss/lss_combmask.dr72.fits")

for i in range(N_MOCKS):
    if use[i]:

        print("Working on sample %d" % i)

        if POSTERIOR:
            model.param_dict = param_dict_sample[i]

        try:
            model.mock.populate(seed=i, Num_ptcl_requirement=300)
        except AttributeError:
            model.populate_mock(halocat, seed=i, Num_ptcl_requirement=300)

        if PROFILE == 'subhalo':
            model.mock.galaxy_table['halo_id'] = (
                    model.mock.galaxy_table['halo_hostid'])

        catalog = utils.get_mock_sdss_catalog(
            model, [window, survey_mask, combmask, combmask_info], seed=i)

        mask = ((catalog['luminosity'] > 10**utils.log_lum_min) &
                (np.logical_or((catalog['z_spec'] < utils.z_max + 0.02) &
                               (catalog['z_spec'] > utils.z_min - 0.02),
                               catalog['collision'])))

        if not TRIM:
            mask = np.ones(len(catalog), dtype=np.bool)

        utils.write_catalog_to_file(catalog[mask], OUTPUT + 'mock_%d.hdf5' % i)
