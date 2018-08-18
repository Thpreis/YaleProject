import pymangle
from astropy.io import fits
import numpy as np

from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import TrivialPhaseSpace, SubhaloPhaseSpace
from halotools.empirical_models import HodModelFactory

from satellite_kinematics import utils

halocat = CachedHaloCatalog(simname='smdpl', redshift=utils.z_halo,
                            dz_tol=0.01)

window = pymangle.Mangle("../sdss/window_pix.dr72bright0.ply")
survey_mask = pymangle.Mangle("../sdss/mask_pix.dr72bright0.ply")
combmask = pymangle.Mangle("../sdss/lss_combmask_pix.dr72.ply")
combmask_info = fits.getdata("../sdss/lss_combmask.dr72.fits")

for i in range(106):

    if i < 3:
        model = utils.default_model(threshold=9.5, profile=(i + 1))
    else:
        cens_occ_model = utils.Lange18Cens(
            threshold=9.0, redshift=utils.z_halo)
        cens_occ_model._suppress_repeated_param_warning = True
        cens_prof_model = TrivialPhaseSpace(redshift=utils.z_halo)
        sats_occ_model = utils.Lange18Sats(
            threshold=9.0, redshift=utils.z_halo)

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

    model.populate_mock(halocat, seed=(i + 100000))

    if i >= 3:
        model.mock.galaxy_table['halo_id'] = (
            model.mock.galaxy_table['halo_hostid'])

    catalog = utils.get_mock_sdss_catalog(
        model, [window, survey_mask, combmask, combmask_info],
        seed=(i + 100000))

    mask = ((catalog['luminosity'] > 10**9.5) &
            (np.logical_or((catalog['z_spec'] < utils.z_max + 0.02) &
                           (catalog['z_spec'] > utils.z_min - 0.02),
                           catalog['collision'])))

    utils.write_catalog_to_file(catalog[mask], 'sdss/mock_%d.hdf5' % i)
