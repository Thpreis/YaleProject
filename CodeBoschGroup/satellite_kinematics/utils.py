import numpy as np

import datetime

import h5py

from scipy.interpolate import interp1d
from scipy.optimize import fminbound, minimize
from scipy import constants

from astropy.cosmology import Planck13
from astropy.table import Table, vstack
from astropy.utils.misc import NumpyRNGContext

from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import TrivialPhaseSpace
from halotools.mock_observables import return_xyz_formatted_array
from halotools.utils import group_member_generator, crossmatch

from .vdbosch04_pair_finder import vdbosch04_pair_finder
from .lange18_components import Lange18Cens, Lange18Sats
from .biased_generalized_nfw_phase_space import BiasedGeneralizedNFWPhaseSpace

try:
    from scipy.spatial import cKDTree as KDT
except ImportError:
    from scipy.spatial import KDTree as KDT
from pyspherematch import spherematch, _spherical_to_cartesian

import warnings

root_directory = '/home/fas/vandenbosch/jl2485/scratch/Satellite_Kinematics'
lib_directory = root_directory + '/lib/satellite_kinematics/'

rp_min = 0.06
z_min = 0.02
z_max = 0.067
a_r = 2.23
b_r = 0.38
c_r = 0.29
a_b = 2.11
b_b = 0.46
c_b = -0.16
x_r_s = 0.15
z_halo = 0.046
cosmology = Planck13
log_lum_min = 9.5

default_log_lum_bins = np.linspace(9.5, 11.0, 11)


def k_correct(z):
    """
    Function returns a fit to the average k-correction as a function of
    redshift z. The k-correction is defined such that
    :math:`m = M + 5 log_{10} [d_L/10 pc] + k.`,
    where :math:`m` is the apparent magnitude, :math:`d_L` the luminosity
    distance and :math:`M` the intrinsic, k-corrected absolute magnitude.

    Parameters
    ----------
    z : array_like
        Array storing the redshifts.

    Returns
    -------
    k : array_like
        Array storing the fit to the average k-correction for the input
        redshifts.
    """

    return (z - 0.1) * 0.8446 + (-0.1008)


def ke_correct(z):
    """
    Function returns a fit to the average k-correction plus evolution
    correction as a function of redshift z. This correction is defined such
    that
    :math:`m = M + 5 log_{10} [d_L/10 pc] + ke.`,
    where :math:`m` is the apparent magnitude, :math:`d_L` the luminosity
    distance and :math:`M` the intrinsic, k-corrected absolute magnitude.

    Parameters
    ----------
    z : array_like
        Array storing the redshifts.

    Returns
    -------
    ke : array_like
        Array storing the fit to the average k and evolution correction for the
        input redshifts.
    """

    q0 = 2.0
    q1 = -1.0
    ecorrect = - (q0 * (1 + q1 * (z - 0.1)) * (z - 0.1))

    kcorrect = k_correct(z)

    return ecorrect + kcorrect


def cross_product_matrix(vector):
    """
    Function returns the cross product matrix of a 3-dimensional vector.

    Parameters
    ----------
    vector : array_like
        Array storing the 3-dimensional vector.

    Returns
    -------
    matrix : np.array of shape [3,3]
        Cross product matrix of the input vector.
    """

    try:
        n = len(vector)
    except TypeError:
        print("The input ``vector`` to the function ``cross_product_matrix`` "
              "must be an array_like object.")
        raise

    try:
        assert n == 3
    except AssertionError:
        print("The input ``vector`` to the function ``cross_product_matrix`` "
              "must have length 3.")
        raise

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def rotation_matrix(vector, theta):
    """
    Function returns a rotation matrix that rotates 3-dimensional vectors
    around the input vector with an angle theta.

    Parameters
    ----------
    vector : array_like
        Array storing the 3-dimensional vector for the rotation matrix. The
        vector does not need to be normalized.
    theta: float
        Angle in radians describing the rotation.

    Returns
    -------
    matrix : np.array of shape [3,3]
        Rotation matrix of the input vector and rotation angle theta.
    """

    try:
        n = len(vector)
    except TypeError:
        print("The input ``vector`` to the function ``rotation_matrix`` "
              "must be an array_like object.")
        raise

    try:
        assert n == 3
    except AssertionError:
        print("The input ``vector`` to the function ``rotation_matrix`` "
              "must have length 3.")
        raise

    try:
        assert theta >= 0
        assert theta <= 2 * np.pi
    except AssertionError:
        print("The input ``theta`` to the function ``rotation_matrix`` "
              "must be a float between 0 and 2*pi.")
        raise

    u = vector / np.sqrt(np.sum(vector**2))
    m = (np.identity(3) * np.cos(theta) +
         np.outer(u, u) * (1 - np.cos(theta)) +
         cross_product_matrix(u) * np.sin(theta))
    return m


def cdf_theta(theta):
    """
    Function returns the CDF of rotation angles that is needed such that we get
    uniform random rotation matrices. See
    https://en.wikipedia.org/wiki/Rotation_matrix for details.

    Parameters
    ----------
    theta: array_like
        Angle in radians describing the rotation.

    Returns
    -------
    cdf : array_like
        CDF value(s) for the input rotation angle(s).
    """

    try:
        assert np.all(np.atleast_1d(theta) >= 0)
        assert np.all(np.atleast_1d(theta) <= np.pi)
    except AssertionError:
        print("The input ``theta`` to the function ``rotation_matrix`` "
              "must be a float or an array with all values between 0 and pi.")
        raise

    return (theta - np.sin(theta)) / np.pi


def random_rotation_matrix(seed=None):
    """
    Function returns a random uniform 3-dimensional rotation matrix.

    Parameters
    ----------
    seed: int, optional
        Seed used to compute the matrix.

    Returns
    -------
    m : np.array of shape [3,3]
        A random uniform 3-dimensional rotation matrix.
    """

    with NumpyRNGContext(seed):
        while True:
            vector = 2 * np.random.random_sample(size=3) - 1
            if np.sum(vector**2) < 1:
                break
        theta = np.linspace(0, np.pi, 1000)
        cdf = cdf_theta(theta)
        theta = (interp1d(cdf, theta))(np.random.random())
        return rotation_matrix(vector, theta)


def default_model(threshold=9.5, profile=1):
    """
    Function returns a halotools HodMockFactory for the default model used in
    this project. The occupation of halos with galaxies is determined by a
    CLF-like empirical model derived from the Cacciato+09 CLF model of
    halotools. Contrary to the Cacciato+09 model, galaxies are also assigned a
    color. The average luminosities of centrals are separately parametrized for
    red and blue galaxies. On the other hand, satellites are all assumed to be
    red. Centrals are assumed to be at rest with respect to the host halo and
    satellites follow a biased NFW distribution.

    Parameters
    ----------
    threshold: float, optional
        Logarithm of the minimum luminosity used to populate dark matter halos.

    Returns
    -------
    model : halotools.empirical_models.HodMockFactory
        The default model used in this project.
    """

    try:
        assert profile == 1 or profile == 2 or profile == 3
    except AssertionError:
        print("The input ``profile`` to the function ``default_model`` "
              "must be 1, 2 or 3.")
        raise

    cens_occ_model = Lange18Cens(threshold=threshold, redshift=z_halo)
    cens_occ_model._suppress_repeated_param_warning = True
    cens_prof_model = TrivialPhaseSpace(redshift=z_halo)
    sats_occ_model = Lange18Sats(threshold=threshold, redshift=z_halo)
    if profile == 1:
        sats_prof_model = BiasedGeneralizedNFWPhaseSpace(
            redshift=z_halo, conc_gal_bias_bins=np.array([1.0]), gamma=1.0,
            cosmology=cosmology)
    elif profile == 2:
        sats_prof_model = BiasedGeneralizedNFWPhaseSpace(
            redshift=z_halo, conc_gal_bias_bins=np.array([1.0 / 2.0]),
            gamma=1.0, cosmology=cosmology)
    elif profile == 3:
        sats_prof_model = BiasedGeneralizedNFWPhaseSpace(
            redshift=z_halo, conc_gal_bias_bins=np.array([1.0 / 2.5]),
            gamma=0.0, cosmology=cosmology)

    model = HodModelFactory(centrals_occupation=cens_occ_model,
                            centrals_profile=cens_prof_model,
                            satellites_occupation=sats_occ_model,
                            satellites_profile=sats_prof_model)

    if profile == 1:
        model.param_dict['conc_gal_bias'] = 1.0
    elif profile == 2:
        model.param_dict['conc_gal_bias'] = 1.0 / 2.0
    else:
        model.param_dict['conc_gal_bias'] = 1.0 / 2.5

    return model


def sigma_200(luminosity, color):
    """
    Function returns the :math:`\sigma_{200}` values as a function of
    luminosity and color given in More et al. (2011).

    Parameters
    ----------
    luminosity: array_like
        Numpy array containing the luminosities of galaxies.
    color: array_like
        Numpy array containing the colors of galaxies.

    Returns
    -------
    s_200 : array_like
        :math:`\sigma_{200}` values as a function of luminosity and color.
    """

    try:
        n_l = len(luminosity)
        n_c = len(color)
    except TypeError:
        n_l = 1
        n_c = 1

    try:
        assert n_l == n_c
    except AssertionError:
        print("The input ``luminosity`` and ``color`` to the function "
              "``sigma_200`` must be arrays of the same length.")
        raise

    try:
        assert np.all(np.logical_or(color == 'red', color == 'blue'))
    except AssertionError:
        print("The input ``color`` to the function ``sigma_200`` "
              "must be a an array of strings containing ``red`` and ``blue``.")

    s_200_r = 10**(a_r + b_r * np.log10(luminosity / 1e10) + c_r *
                   np.log10(luminosity / 1e10)**2) / 200.0
    s_200_b = 10**(a_b + b_b * np.log10(luminosity / 1e10) + c_b *
                   np.log10(luminosity / 1e10)**2) / 200.0
    s_200 = np.where(color == 'red', s_200_r, s_200_b)

    s_200 = np.where(s_200 > 5.0, 5.0, s_200)

    return s_200


def read_catalog_from_file(filename, fiber_collisions=True, z_spec_min=0.01):
    """
    Function reads a spectroscopic galaxy catalog from a file.

    Parameters
    ----------
    filename: string
        The location of the file.
    fiber_collisions: bool, optional
        Boolean describing whether fiber collisions should be included in the
        catalog. This option is only available for mock catalogs where the true
        redshifts are known.
    z_spec_min: float, optional
        The minimum redshift of galaxies in the returned galaxy catalog.

    Returns
    -------
    catalog : astropy.table.Table
        Astropy table containing the spectroscopic galaxy catalog. The table
        contains

        * a galaxy ID ('gal_id'),
        * the spectroscopic redshift ('z_spec'),
        * the absolute luminosity in solar luminosity units ('luminosity'),
        * the angular coordinates in degrees ('ra' and 'dec'),
        * the comoving distance corresponding to the spectroscopic redshift\
            ('d_com'),
        * the color of the galaxy ('color'),
        * the spectroscopic completeness from SDSS corresponding to the\
            angular coordinates ('f_spec'),
        * the angular completeness defined as the fraction of the area\
            determined by r_h inside the survey window function and outside\
            the survey mask ('completeness'),
        * the spectroscopic weight used to correct fiber collision effects\
            ('weight_spec')
        * and whether a fiber collision occurred ('collision'). If a fiber\
            collision occurred, the galaxy will be assigned the redshift of\
            the nearest neighbour. If fiber collisions are turned off, the\
            spectroscopic weights are appropriately set to unity.

        Additionally, if the catalog was created from a simulation it will
        additionally contain

        * a halo ID ('halo_id'),
        * a host halo mass ('halo_mvir') and
        * a galaxy type ('gal_type') associated with each galaxy.
    """

    catalog = Table.read(filename, path='catalog')

    catalog['color'] = np.where(catalog['red'], 'red', 'blue')
    catalog.remove_column('red')

    if 'central' in catalog.colnames:
        catalog['gal_type'] = np.where(catalog['central'], 'centrals',
                                       'satellites')
        catalog.remove_column('central')

    if not fiber_collisions:
        try:
            catalog['z_spec'][:] = catalog['redshift']
        except KeyError:
            print("The spectroscopic catalog that was read does not contain "
                  "information about intrinsic redshift. Thus, the effect of "
                  "fiber collisions cannot be reversed.")
            raise

        catalog['z_spec'][:] = catalog['redshift']
        catalog['d_com'][:] = (
            cosmology.comoving_distance(catalog['z_spec']).value *
            cosmology.H0.value / 100.0)
        catalog['collision'][:] = False
        catalog['weight_spec'][:] = 1.0

    if 'redshift' in catalog.colnames:
        catalog.remove_column('redshift')

    return catalog[catalog['z_spec'] >= z_spec_min]


def write_catalog_to_file(catalog, fname, overwrite=True):

    if 'gal_id' not in catalog.colnames:
        catalog['gal_id'] = np.arange(len(catalog))
    catalog['red'] = catalog['color'] == 'red'

    if 'halo_id' in catalog.colnames:
        catalog['central'] = catalog['gal_type'] == 'centrals'
        catalog.keep_columns([
            'gal_id', 'halo_id', 'z_spec', 'collision', 'bhg', 'redshift',
            'ra', 'dec', 'luminosity', 'd_com', 'halo_mvir', 'f_spec',
            'completeness', 'weight_spec', 'red', 'central'])

    else:
        catalog.keep_columns([
            'gal_id', 'z_spec', 'collision', 'ra', 'dec', 'luminosity',
            'd_com', 'f_spec', 'completeness', 'weight_spec', 'red'])

    catalog.write(fname, path='catalog', overwrite=overwrite)


def vdbosch04_pair_finder_sphere(sample, r_h, z_h, r_s, z_s, marks,
                                 mark_min=0):
    """
    Function for finding pairs according to vdBosch et al. 2004. Contrary to
    vdbosch04_pair_finder, this function works on a sphere.

    Points are considered primaries if
        1. they contain no point with a higher mark within a cylindrical volume
        described by r_h and z_h and
        2. they are not contained within the cylindrical volume given by r_h
        and z_h of another point of higher mark that is considered a central.

    Points are considered secondaries if they lie within a cylindrical volume
    of a point that is considered a primary. The volume is defined by r_s and
    z_s of the central. Primaries and their secondaries form a pair.

    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing positions of points. The first
        coordinate is right ascension in degrees, the second coordinate the
        declination in degrees and the third the comoving distance.

    r_h : array_like
        Radius of the cylinder to search for neighbors around points. If a
        single float is given, ``r_h`` is assumed to be the same for each point
        in the ``sample``.

    z_h : array_like
        Half the length of cylinders to search for neighbors around points. If
        a single float is given, ``z_h`` is assumed to be the same for each
        point in the ``sample``.

    r_s : array_like
        Radius of the cylinder to search for satellites around centrals. If a
        single float is given, ``r_h`` is assumed to be the same for each point
        in the ``sample``.

    z_s : array_like
        Half the length of cylinders to search for satellites around centrals.
        If a single float is given, ``z_h`` is assumed to be the same for each
        point in the ``sample``.

    marks : array_like
        *Npts* array of marks.

    mark_min : float, optional
        Minimum mark for a point to be considered a primary. Using this can
        considerably speed up computations.

    Returns
    -------
    prim : numpy.array
        Int array containing the indices of primaries in primary-secondary
        pairs.

    secd : numpy.array
        Int array containing the indices of secondaries in primary-secondary
        pairs. Every primary is also considered a secondary of itself.
    """

    sample_rad = np.copy(sample)

    sample_rad[:, 0] = sample[:, 0] * np.pi / 180.0
    sample_rad[:, 1] = sample[:, 1] * np.pi / 180.0 + np.pi / 2.0

    ell = 1.0 / np.sin(sample_rad[:, 1])
    r_h = r_h / sample_rad[:, 2]
    r_s = r_s / sample_rad[:, 2]

    period = [2 * np.pi, np.pi + 2 * max(np.max(r_h), np.max(r_s)),
              np.max(sample[:, 2]) + 2 * np.max(z_h)]
    approx_cell_size = np.array([0.2, 0.1, 50])

    return vdbosch04_pair_finder(
        sample_rad, r_h, z_h, r_s, z_s, marks, ell=ell, period=period,
        approx_cell_size=approx_cell_size, mark_min=mark_min)


def cylinder_size(luminosity, color, redshift=None):
    """
    Function computes the cylinder sizes around galaxies of a given luminosity
    and color.

    Parameters
    ----------
    luminosity : array_like
        The luminosities of the galaxies.

    color : array_like
        The different colors of the galaxies.

    redshift : float, optional
        The redshift of each galaxy. This is needed because the cylinder sizes
        are given in comoving distances but actually correspond to velocities.
        The default case corresponds to redshift 0.

    Returns
    -------
    r_h : array_like
        The radius of the cylinder used to determine whether a galaxy is a
        primary.

    z_h : array_like
        The height of the cylinder used to determine whether a galaxy is a
        primary.

    r_s : array_like
        The radius of the cylinder used to associate secondaries to primaries.

    z_s : array_like
        The height of the cylinder used to associate secondaries to primaries.
    """

    try:
        n_l = len(luminosity)
        n_c = len(color)
    except TypeError:
        n_l = 1
        n_c = 1

    try:
        assert n_l == n_c
    except AssertionError:
        print("The input ``luminosity`` and ``color`` to the function "
              "``cylinder_size`` must be arrays of the same length.")
        raise

    try:
        assert np.all(np.logical_or(color == 'red', color == 'blue'))
    except AssertionError:
        print("The input ``color`` to the function ``cylinder_size`` "
              "must be a an array of strings containing ``red`` and ``blue``.")

    if redshift is not None:
        try:
            n_z = len(redshift)
        except TypeError:
            n_z = 1

        try:
            assert n_z == n_l or n_z == 1
        except AssertionError:
            print("The input ``redshift`` to the function ``cylinder_size`` "
                  "must be a float or an array of the same length as the "
                  "input ``luminosity``.")

    r_h = 0.5 * sigma_200(luminosity, np.repeat('red', len(color)))
    z_h = 1000 * sigma_200(luminosity, np.repeat('red', len(color))) / 100.0
    r_s = x_r_s * sigma_200(luminosity, color)
    z_s = np.ones(len(luminosity)) * 4100 / 100.0

    if redshift is not None:
        x = (1 + redshift) / np.sqrt(cosmology.Om0 * (1 + redshift)**3.0 +
                                     1 - cosmology.Om0)
        z_h *= x
        z_s *= x

    return r_h, z_h, r_s, z_s


def pairs_from_catalog(catalog, period=None):
    """
    Function determines primary-secondary pairs and their associated
    properties from a catalog.

    Parameters
    ----------
    catalog : astropy.table.Table
        An astropy table containing the galaxies.

    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions in each
        dimension. This only needs to be set if the catalog is for galaxies
        populating a simulation box instead of a sky survey.

    Returns
    -------
    pairs : astropy.table.Table
        An astropy table containing the pairs and additional information. The
        table will always contain

        * the galaxy ID of the primary ('gal_id_prim'),
        * the absolute luminosity of the primary in solar luminosity units\
            ('lum_prim'),
        * the color of the primary ('color_prim'),
        * whether the pair consists of only the primary ('prim_pair'),
        * the projected distance between the primary and the secondary ('rp'),
        * the line-of-sight velocity difference between primary and secondary\
            ('vz') and
        * whether the secondary should be used to compute the constraints\
            ('use_vz').

        Additionally, if the input galaxy catalog is a sky survey, it will also
        contain

        * the angular coordinates of the primary in degrees ('ra_prim' and\
            'dec_prim'),
        * the spectroscopic weight of the primary used to correct fiber\
        collision effects ('weight_spec_prim')
        * the spectroscopic weight of the secondary used to correct fiber\
        collision effects ('weight_spec_secd')
        * the spectroscopic redshift of the primary ('z_spec_prim') and
        * the comoving distance corresponding to the spectroscopic redshift\
            ('d_com_prim').

        Additionally, if the catalog was created from a simulation it will
        additionally contain

        * the halo ID of the halo the primary resides in ('halo_id_prim'),
        * the halo mass of the halo the primary resides in ('halo_mvir_prim'),
        * whether primary and secondary are not hosted by the same dark matter\
            halo ('interloper'),
        * whether the primary is the brightest halo galaxy of its dark matter\
            halo ('bhg') and
        * the galaxy type of the primary ('gal_type_prim').
    """

    pairs = Table()

    if 'ra' in catalog.colnames and 'dec' in catalog.colnames:
        sdss_like = True

        colnames = ['z_spec', 'd_com', 'f_spec', 'completeness', 'collision',
                    'weight_spec', 'color', 'luminosity']
        for colname in colnames:
            try:
                assert colname in catalog.colnames
            except AssertionError:
                print("The input ``catalog`` must contain the column ``%s``."
                      % colname)
                raise

    else:
        sdss_like = False

        period = np.atleast_1d(period)
        if len(period) == 1:
            period = np.repeat(period, 3)

        try:
            assert period is not None
        except AssertionError:
            print("If the input ``catalog`` to the pairs from catalog "
                  "function is not a sky survey, i.e. contains angular "
                  "coordinates, the input ``period`` must be given.")
            raise

        colnames = ['x', 'y', 'z', 'vz', 'color', 'luminosity']
        for colname in colnames:
            try:
                assert colname in catalog.colnames
            except AssertionError:
                print("The input ``catalog`` must contain the column ``%s``."
                      % colname)
                raise

        try:
            assert np.all(catalog['x'] >= 0)
            assert np.all(catalog['x'] <= period[0])
            assert np.all(catalog['y'] >= 0)
            assert np.all(catalog['y'] <= period[1])
            assert np.all(catalog['z'] >= 0)
            assert np.all(catalog['z'] <= period[2])
        except AssertionError:
            print("All input x, y and z coordinates must be inside the "
                  "periodic boundary conditions.")
            raise

    if 'halo_id' in catalog.colnames and 'bhg' not in catalog.colnames:
        grouping_key = 'halo_id'
        requested_columns = ['luminosity']
        catalog.sort(grouping_key)
        group_gen = group_member_generator(catalog, grouping_key,
                                           requested_columns)

        catalog['bhg'] = np.zeros(len(catalog), dtype=np.bool)
        for first, last, member_props in group_gen:
            bhg = np.argmax(member_props[0])
            catalog['bhg'][first+bhg] = True

    if sdss_like:
        redshift = catalog['z_spec']
        pos = return_xyz_formatted_array(
            catalog['ra'], catalog['dec'], catalog['d_com'])
    else:
        redshift = None
        pos = return_xyz_formatted_array(
            catalog['x'], catalog['y'], catalog['z'], velocity=catalog['vz'],
            period=period, velocity_distortion_dimension='z')

    r_h, z_h, r_s, z_s = cylinder_size(catalog['luminosity'], catalog['color'],
                                       redshift=redshift)

    if sdss_like:
        prim, secd = vdbosch04_pair_finder_sphere(pos, r_h, z_h, r_s, z_s,
                                                  catalog['luminosity'])

        mask = ((catalog['f_spec'][prim] > 0.8) &
                (catalog['completeness'][prim] > 0.8) &
                (redshift[prim] >= z_min) & (redshift[prim] <= z_max) &
                np.logical_not(catalog['collision'][prim]) &
                np.logical_not(catalog['collision'][secd]))
        prim = prim[mask]
        secd = secd[mask]

    else:
        prim, secd = vdbosch04_pair_finder(
            pos, r_h, z_h, r_s, z_s, catalog['luminosity'], period=period)

    secd = secd[np.argsort(prim)]
    prim = prim[np.argsort(prim)]

    pairs['gal_id_prim'] = prim
    pairs['gal_id_secd'] = secd
    pairs['lum_prim'] = catalog['luminosity'][prim]
    pairs['lum_secd'] = catalog['luminosity'][secd]
    pairs['color_prim'] = catalog['color'][prim]
    pairs['prim_pair'] = (prim == secd)

    if sdss_like:
        pairs['rp'] = (catalog['d_com'][prim] * np.sqrt(
            ((catalog['ra'][prim] - catalog['ra'][secd]) *
             np.sin(catalog['dec'][prim] * np.pi / 180.0 + np.pi / 2.0))**2 +
            (catalog['dec'][prim] - catalog['dec'][secd])**2) * np.pi / 180.0)
        vz_prim = (redshift[prim] * (constants.c * 1e-3) / (1.0 +
                                                            redshift[prim]))
        vz_secd = (redshift[secd] * (constants.c * 1e-3) / (1.0 +
                                                            redshift[prim]))
        pairs['vz'] = vz_secd - vz_prim
        pairs['ra_prim'] = catalog['ra'][prim]
        pairs['dec_prim'] = catalog['dec'][prim]
        pairs['weight_spec_prim'] = catalog['weight_spec'][prim]
        pairs['weight_spec_secd'] = catalog['weight_spec'][secd]
        pairs['z_spec_prim'] = redshift[prim]
        pairs['d_com_prim'] = catalog['d_com'][prim]

    else:
        dx = np.abs(catalog['x'][prim] - catalog['x'][secd])
        dx = np.minimum(dx, period[0] - dx)
        dy = np.abs(catalog['y'][prim] - catalog['y'][secd])
        dy = np.minimum(dy, period[1] - dy)
        pairs['rp'] = np.array(np.sqrt(dx**2 + dy**2), dtype=np.float64)

        dvz = pos[:, 2][prim] - pos[:, 2][secd]
        dvz = np.where(dvz < -period[2] / 2.0, dvz + period[2], dvz)
        dvz = np.where(dvz > period[2] / 2.0, dvz - period[2], dvz)
        dvz *= 100
        pairs['vz'] = np.array(dvz, dtype=np.float64)

    if 'halo_mvir' in catalog.colnames:
        pairs['halo_mvir_prim'] = catalog['halo_mvir'][prim]

    if 'halo_id' in catalog.colnames:
        pairs['interloper'] = np.logical_not(catalog['halo_id'][prim] ==
                                             catalog['halo_id'][secd])
        pairs['halo_id_prim'] = catalog['halo_id'][prim]

    if 'bhg' in catalog.colnames:
        pairs['bhg'] = catalog['bhg'][prim]

    if 'gal_type' in catalog.colnames:
        pairs['gal_type_prim'] = catalog['gal_type'][prim]

    pairs['use_vz'] = ((prim != secd) & (pairs['rp'] > rp_min))
    pairs = pairs[np.abs(pairs['vz']) <= 4000]

    return pairs


class RedshiftSpaceModel:

    def __init__(self, gal_type, profile=1):

        self.gal_type = gal_type

        if gal_type == 'interloper':
            self.rp_bins = np.linspace(0, 2, 100)
            self.vz_bins = np.linspace(0, 4000, 400)
            self.d_n = np.repeat(np.diff(self.rp_bins**2),
                                 len(self.vz_bins) - 1).reshape(
                    (len(self.rp_bins) - 1, len(self.vz_bins) - 1))
            self.log_mvir = np.nan

            # calculates the normalization
            norm = np.cumsum(np.sum(
                np.outer(np.diff(self.rp_bins), np.diff(self.vz_bins)) *
                self.d_n, axis=1))
            norm = np.concatenate([[0], norm])
            self.norm = interp1d(
                self.rp_bins, norm - np.interp(rp_min, self.rp_bins, norm))

        elif gal_type == 'member':

            try:
                assert profile == 1 or profile == 2 or profile == 3
            except AssertionError:
                print("The input ``profile`` must be 1, 2 or 3.")
                raise

            fname = lib_directory + ('lookup/template_member_%d.hdf5' %
                                     profile)
            self.log_mvir = np.linspace(11.0, 15.0, 41)

            f = h5py.File(fname, 'r')

            self.rp_bins = f['rp_bins'][:]
            self.vz_bins = f['vz_bins'][:]

            self.d_n = np.zeros([len(self.log_mvir), len(self.rp_bins) - 1,
                                 len(self.vz_bins) - 1])
            self.norm = []

            # calculates the normalization or fraction inside aperture
            for i, log_mvir in enumerate(self.log_mvir):
                self.d_n[i] = f['log_mvir=%.1f' % log_mvir][:]
                norm = np.cumsum(
                    np.sum(self.d_n[i] * np.outer(np.diff(self.rp_bins),
                                                  np.diff(self.vz_bins)),
                           axis=1))
                norm = np.concatenate([[0], norm])
                self.norm.append(interp1d(
                    self.rp_bins,
                    norm - np.interp(rp_min, self.rp_bins, norm)))

            f.close()

    def probability_density(self, rp, vz, rp_max=2.0):

        try:
            assert isinstance(rp, np.ndarray)
            assert isinstance(vz, np.ndarray)
            assert len(rp) == len(vz)
        except AssertionError:
            raise ValueError('Input rp and vz must be arrays of the same ' +
                             'length.')

        rp_max = np.atleast_1d(rp_max)
        if len(rp_max) == 1:
            rp_max = np.repeat(rp_max, len(rp))

        try:
            assert np.all(rp_max <= 2.0)
            assert np.all(rp_max > rp_min)
        except:
            raise ValueError('Input rp_max must be in [%.2f, 2.0].' % rp_min)

        try:
            assert np.all(rp <= rp_max)
            assert np.all(np.abs(vz) <= 4000.0)
        except:
            raise ValueError('Input rp and vz must be in [0.0, rp_max] '
                             'and [-4000, 4000], respectively.')

        vz_dig = np.digitize(vz, self.vz_bins, right=True) - 1
        rp_dig = np.digitize(rp, self.rp_bins, right=True) - 1

        if self.gal_type == 'member':
            p_mem = np.zeros([len(self.log_mvir), len(vz)])
            for i in range(len(self.log_mvir)):
                p_mem[i] = self.d_n[i, rp_dig, vz_dig] / self.norm[i](rp_max)
            return p_mem
        else:
            return self.d_n[rp_dig, vz_dig] / self.norm(rp_max)

    def f_ap(self, rp_max):

        try:
            assert np.all(rp_max <= 2.0)
            assert np.all(rp_max > rp_min)
        except:
            raise ValueError('Input rp_max must be in [%.2f, 2.0].' % rp_min)

        if self.gal_type == 'member':
            norm = np.zeros([len(self.log_mvir), len(rp_max)])
            for i in range(len(self.log_mvir)):
                norm[i] = self.norm[i](rp_max)
            return norm
        else:
            return self.norm(rp_max)


def get_p_mem_tot(log_mvir, sigma, p_mem, log_mvir_bins, f_ap):

    log_mvir = min(log_mvir, 15.0)
    log_mvir = max(log_mvir, 10.0)

    sigma = min(sigma, 1.5)
    sigma = max(sigma, 0.1)

    log_mvir_weights = np.exp(- (log_mvir - log_mvir_bins) ** 2 /
                                (2.0 * sigma**2)) * f_ap.T
    log_mvir_weights /= np.sum(log_mvir_weights, axis=1)[:, np.newaxis]

    return np.sum(p_mem.T * log_mvir_weights, axis=1)


def fit_interloper_fraction(theta, rp, vz, weights, p_mem, log_mvir_bins,
                            f_ap, p_int, return_likelihood_only=False):

    log_mvir = theta[0]
    sigma = theta[1]

    p_mem_tot = get_p_mem_tot(log_mvir, sigma, p_mem, log_mvir_bins, f_ap)

    neg_log_lik = lambda f_int: - np.sum(
        np.log(p_mem_tot * (1 - f_int) + f_int * p_int) * weights)
    res = fminbound(neg_log_lik, 0, 1, full_output=True)

    if return_likelihood_only:
        return res[1]

    return res[0], res[1]


def membership_probability(rp, vz, weights, model_mem, model_int, rp_max):

    p_mem = model_mem.probability_density(rp, np.abs(vz), rp_max=rp_max)
    log_mvir_bins = model_mem.log_mvir
    f_ap = model_mem.f_ap(rp_max)
    p_int = model_int.probability_density(rp, np.abs(vz), rp_max=rp_max)

    fit = minimize(
        fit_interloper_fraction, x0=[13.0, 0.5], method='Nelder-Mead',
        args=(rp, vz, weights, p_mem, log_mvir_bins, f_ap, p_int, True),
        options={'maxiter': 400})

    log_mvir = fit.x[0]
    sigma = fit.x[1]

    p_mem_tot = get_p_mem_tot(log_mvir, sigma, p_mem, log_mvir_bins, f_ap)
    f_int = fit_interloper_fraction(
        fit.x, rp, vz, weights, p_mem, log_mvir_bins, f_ap, p_int)[0]

    return p_mem_tot * (1 - f_int) / (p_mem_tot * (1 - f_int) + p_int * f_int)


def get_membership_probability(pairs, profile=1, use_spec_weights=True,
                               log_lum_bins=default_log_lum_bins):

    model_int = RedshiftSpaceModel('interloper')
    model_mem = RedshiftSpaceModel('member', profile=profile)

    if use_spec_weights and 'weight_spec_prim' in pairs.colnames:
        pairs['w_sw'] = pairs['weight_spec_prim'] * pairs['weight_spec_secd']
        pairs['w_hw'] = pairs['weight_spec_prim'] * pairs['weight_spec_secd']
    else:
        pairs['w_sw'] = np.ones(len(pairs))
        pairs['w_hw'] = np.ones(len(pairs))

    pairs['p_member_sw'] = np.zeros(len(pairs))
    pairs['p_member_hw'] = np.zeros(len(pairs))

    grouping_key = 'gal_id_prim'
    requested_columns = ['use_vz']
    pairs.sort(grouping_key)
    group_gen = group_member_generator(pairs, grouping_key, requested_columns)

    for first, last, member_props in group_gen:
        use_vz = member_props[0]
        if np.any(use_vz):
            pairs['w_hw'][first:last] /= np.sum(use_vz)

    for i in range(len(log_lum_bins) - 1):
        for color in ['red', 'blue']:

            rp_max = cylinder_size(pairs['lum_prim'], pairs['color_prim'],
                                   redshift=None)[2]

            mask = ((log_lum_bins[i] <= np.log10(pairs['lum_prim'])) &
                    (np.log10(pairs['lum_prim']) < log_lum_bins[i + 1]) &
                    (pairs['rp'] > rp_min) & (pairs['rp'] < rp_max) &
                    (pairs['color_prim'] == color))

            if np.any(mask):
                pairs['p_member_sw'][mask] = membership_probability(
                    pairs['rp'][mask], pairs['vz'][mask], pairs['w_sw'][mask],
                    model_mem, model_int, rp_max[mask])
                pairs['p_member_hw'][mask] = membership_probability(
                    pairs['rp'][mask], pairs['vz'][mask], pairs['w_hw'][mask],
                    model_mem, model_int, rp_max[mask])

    return pairs


def get_mock_sdss_catalog(model, mangle, seed=None, rotation=True,
                          no_bnc=True):

    window = mangle[0]
    mask = mangle[1]
    combmask = mangle[2]
    combmask_info = mangle[3]

    z = np.linspace(0.0, 1.0, 100000)
    distmod = interp1d(z, cosmology.distmod(z).value)

    comoving_distance = (cosmology.comoving_distance(z).value *
                         cosmology.H0.value / 100.0)
    z = interp1d(comoving_distance, z)

    comoving_distance_max = (cosmology.comoving_distance(0.2).value *
                             cosmology.H0.value / 100.0)
    n_catalog_repeat = int(np.ceil(comoving_distance_max / model.mock.Lbox[0] -
                                   0.5))

    gals = model.mock.galaxy_table['x', 'y', 'z', 'vx', 'vy', 'vz',
                                   'luminosity', 'halo_id', 'gal_type',
                                   'halo_mvir', 'color']

    if no_bnc:
        gals = remove_bnc(gals)

    gals['neg_luminosity'] = -gals['luminosity']
    gals.sort(['halo_id', 'neg_luminosity'])
    gals['bhg'] = np.zeros(len(gals), dtype=np.bool)
    gals['bhg'][np.unique(gals['halo_id'], return_index=True)[1]] = True
    del gals['neg_luminosity']

    with NumpyRNGContext(seed):
        x_translation = np.random.random_sample() * model.mock.Lbox[0]
        y_translation = np.random.random_sample() * model.mock.Lbox[1]
        z_translation = np.random.random_sample() * model.mock.Lbox[2]
        gals['x'] = ((gals['x'] + x_translation) % model.mock.Lbox[0] -
                     model.mock.Lbox[0] / 2.0)
        gals['y'] = ((gals['y'] + y_translation) % model.mock.Lbox[1] -
                     model.mock.Lbox[1] / 2.0)
        gals['z'] = ((gals['z'] + z_translation) % model.mock.Lbox[2] -
                     model.mock.Lbox[2] / 2.0)

    id_offset = 0
    catalog = gals.copy()
    catalog = catalog[:0]

    for x_cell in range(-n_catalog_repeat, n_catalog_repeat + 1):
        for y_cell in range(-n_catalog_repeat, n_catalog_repeat + 1):
            for z_cell in range(-n_catalog_repeat, n_catalog_repeat + 1):
                gals_shifted = gals.copy()
                gals_shifted['x'] = gals['x'] + model.mock.Lbox[0] * x_cell
                gals_shifted['y'] = gals['y'] + model.mock.Lbox[1] * y_cell
                gals_shifted['z'] = gals['z'] + model.mock.Lbox[2] * z_cell

                gals_shifted['d_com'] = np.sqrt(gals_shifted['x'] ** 2 +
                                                gals_shifted['y'] ** 2 +
                                                gals_shifted['z'] ** 2)
                gals_shifted['redshift'] = z(gals_shifted['d_com'])

                m_rel = (4.76 - 2.5 * np.log10(gals_shifted['luminosity']) +
                         distmod(gals_shifted['redshift']) +
                         5 * np.log10(cosmology.H0.value / 100.0) +
                         ke_correct(gals_shifted['redshift']) - 0.01)

                gals_shifted = gals_shifted[m_rel <= 17.6]

                gals_shifted['halo_id'] += id_offset

                catalog = vstack([catalog, gals_shifted])

                id_offset += 100000000000

    catalog['v_los'] = ((catalog['x'] * catalog['vx'] + catalog['y'] *
                        catalog['vy'] + catalog['z'] * catalog['vz']) /
                        catalog['d_com'])
    with NumpyRNGContext(seed):
        catalog['redshift'] += ((1 + catalog['redshift']) * 1e3 / constants.c *
                                catalog['v_los'])
        catalog['redshift'] += 1e3 / constants.c * np.random.normal(scale=15.)

    if rotation:
        m = random_rotation_matrix(seed)
        x_new = (m[0, 0] * catalog['x'] + m[0, 1] * catalog['y'] +
                 m[0, 2] * catalog['z'])
        y_new = (m[1, 0] * catalog['x'] + m[1, 1] * catalog['y'] +
                 m[1, 2] * catalog['z'])
        z_new = (m[2, 0] * catalog['x'] + m[2, 1] * catalog['y'] +
                 m[2, 2] * catalog['z'])
        catalog['x'], catalog['y'], catalog['z'] = x_new, y_new, z_new

    theta = np.arccos(catalog['z'] / catalog['d_com'])
    phi = np.arctan2(catalog['y'], catalog['x'])

    catalog['dec'] = 90 - theta * 180.0 / np.pi
    catalog['ra'] = np.where(phi >= 0, phi, phi + 2 * np.pi) * 180.0 / np.pi

    catalog = catalog[window.contains(catalog['ra'], catalog['dec'])]
    catalog = catalog[np.logical_not(mask.contains(catalog['ra'],
                                                   catalog['dec']))]

    polyid = combmask.polyid(catalog['ra'], catalog['dec'])
    catalog['f_spec'] = combmask_info['FGOTMAIN'][polyid]

    catalog['completeness'] = np.zeros(len(catalog), dtype=np.float32)

    r = (cylinder_size(catalog['luminosity'], catalog['color'],
                       redshift=catalog['redshift'])[0] /
         catalog['d_com'] * 180 / np.pi)

    catalog['completeness'] = completeness_score(catalog['ra'], catalog['dec'],
                                                 r, window, mask)

    catalog['collision'], catalog['z_spec'] = fiber_collisions(
        catalog['ra'], catalog['dec'], catalog['redshift'], seed=seed)

    catalog['weight_spec'] = 1.0 / get_fiber_collision_probability(catalog)

    catalog['d_com'] = (cosmology.comoving_distance(catalog['z_spec']).value *
                        cosmology.H0.value / 100.0)

    return catalog


def get_fiber_collision_probability(catalog):

    k_max = 3

    idx = np.zeros([k_max, len(catalog)], dtype=np.int32)
    ds = np.zeros([k_max, len(catalog)], dtype=np.float32)
    for k in range(k_max):
        dummy, idx[k], ds[k] = spherematch(
            catalog['ra'], catalog['dec'], catalog['ra'], catalog['dec'],
            nnearest=k + 2)

    ds = ds * 3600
    n_near = np.sum(ds < 55.0, axis=0)
    p_coll = np.zeros(len(catalog))

    for i in range(k_max + 1):
        mask = (n_near == i)
        if np.any(mask):
            p_coll[mask] = 1.0 - np.mean(catalog['collision'][mask])

    return p_coll


def completeness_score(ra, dec, r, window, mask):

    phi_bins = np.linspace(0, 2 * np.pi, 50)

    comp = np.zeros(len(ra))
    w_tot = 0.0

    for phi in phi_bins[:-1]:
        ra_i = ra + np.sin(phi) * r / np.cos(dec * np.pi / 180.0)
        dec_i = dec + np.cos(phi) * r
        comp += (window.contains(ra_i, dec_i) &
                 np.logical_not(mask.contains(ra_i, dec_i)))
        w_tot += 1.0

    comp /= w_tot

    return comp


def constraints_dict_from_spec_catalog(
        catalog, profile=1, log_lum_bins=default_log_lum_bins):

    warnings.simplefilter(action='ignore', category=FutureWarning)

    constraints = {}

    d_com_min = (cosmology.comoving_distance(z_min).value *
                 cosmology.H0.value / 100.0)
    d_com_max = (cosmology.comoving_distance(z_max).value *
                 cosmology.H0.value / 100.0)

    catalog = catalog[(catalog['z_spec'] > 0.01) &
                      (catalog['luminosity'] > 10**log_lum_min)]
    pairs = pairs_from_catalog(catalog)
    pairs = get_membership_probability(pairs, use_spec_weights=True,
                                       profile=profile)

    a_eff = 2.27137
    mask = ((z_min < catalog['z_spec']) & (catalog['z_spec'] < z_max) &
            np.logical_not(catalog['collision']) & (catalog['f_spec'] > 0.8))

    constraints['n_gal'] = np.histogram(
        np.log10(catalog['luminosity'][mask]), bins=log_lum_bins,
        weights=catalog['weight_spec'][mask])[0]
    constraints['n_gal'] /= a_eff / 3.0 * (d_com_max**3 - d_com_min**3)

    for color in ['red', 'blue']:
        constraints['n_mem_' + color[0]] = np.zeros(
            len(log_lum_bins) - 1, dtype=np.float64)
        constraints['sigma_hw_' + color[0]] = np.zeros(
            len(log_lum_bins) - 1, dtype=np.float64)
        constraints['sigma_sw_' + color[0]] = np.zeros(
            len(log_lum_bins) - 1, dtype=np.float64)
    constraints['f_pri_r'] = np.zeros(len(log_lum_bins) - 1, dtype=np.float64)

    for i in range(len(log_lum_bins) - 1):
        lum_min = 10**log_lum_bins[i]
        lum_max = 10**log_lum_bins[i+1]
        mask_prim = ((lum_min <= pairs['lum_prim']) &
                     (pairs['lum_prim'] < lum_max) & pairs['prim_pair'])
        mask_secd = ((lum_min <= pairs['lum_prim']) &
                     (pairs['lum_prim'] < lum_max) & pairs['use_vz'] &
                     (~pairs['prim_pair']))

        for color in ['red', 'blue']:
            mask_prim_tmp = mask_prim & (pairs['color_prim'] == color)
            mask_secd_tmp = mask_secd & (pairs['color_prim'] == color)
            if np.sum(pairs['p_member_hw'][mask_secd_tmp]) > 10:
                constraints['n_mem_' + color[0]][i] = (np.sum(
                    pairs['p_member_sw'][mask_secd_tmp] *
                    pairs['w_sw'][mask_secd_tmp]) /
                    np.sum(pairs['weight_spec_prim'][mask_prim_tmp]))
                for w in ['hw', 'sw']:
                    constraints['sigma_' + w + '_' + color[0]][i] = np.sqrt(
                        np.average(
                            pairs['vz'][mask_secd_tmp]**2,
                            weights=(pairs['p_member_' + w][mask_secd_tmp] *
                                     pairs['w_' + w][mask_secd_tmp])) -
                        2.0 * 15.0**2)
            else:
                constraints['n_mem_' + color[0]][i] = np.nan
                constraints['sigma_hw_' + color[0]][i] = np.nan
                constraints['sigma_sw_' + color[0]][i] = np.nan

        if np.any(mask_prim):
            constraints['f_pri_r'][i] = np.average(
                pairs['color_prim'][mask_prim] == 'red',
                weights=pairs['weight_spec_prim'][mask_prim])
        else:
            constraints['f_pri_r'][i] = np.nan

    constraints['log_sigma_hw_r'] = np.log10(constraints['sigma_hw_r'])
    constraints['log_sigma_hw_b'] = np.log10(constraints['sigma_hw_b'])
    constraints['r_hw_sw_sq_r'] = (constraints['sigma_hw_r'] /
                                   constraints['sigma_sw_r'])**2
    constraints['r_hw_sw_sq_b'] = (constraints['sigma_hw_b'] /
                                   constraints['sigma_sw_b'])**2

    return constraints


def write_constraints_to_file(constraints, fname, path, append=True):

    f = h5py.File(fname, "a")

    f.attrs['fname'] = fname
    f.attrs['ftime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    f.attrs['HDF5_Version'] = h5py.version.hdf5_version
    f.attrs['h5py_version'] = h5py.version.version

    f.close()

    Table(constraints).write(fname, path=path, append=append)


def constraints_dict_to_vector(constraints_dict):

    constraints_vector = np.zeros(80, dtype=constraints_dict['n_gal'][0].dtype)

    constraints_vector[0:10] = constraints_dict['n_gal']
    constraints_vector[10:20] = constraints_dict['f_pri_r']
    constraints_vector[20:30] = constraints_dict['n_mem_r']
    constraints_vector[30:40] = constraints_dict['n_mem_b']
    constraints_vector[40:50] = constraints_dict['log_sigma_hw_r']
    constraints_vector[50:60] = constraints_dict['r_hw_sw_sq_r']
    constraints_vector[60:70] = constraints_dict['log_sigma_hw_b']
    constraints_vector[70:80] = constraints_dict['r_hw_sw_sq_b']

    return constraints_vector


def constraints_vector_to_dict(constraints_vector):

    constraints_dict = {}

    constraints_dict['n_gal'] = constraints_vector[0:10]
    constraints_dict['f_pri_r'] = constraints_vector[10:20]
    constraints_dict['n_mem_r'] = constraints_vector[20:30]
    constraints_dict['n_mem_b'] = constraints_vector[30:40]
    constraints_dict['log_sigma_hw_r'] = constraints_vector[40:50]
    constraints_dict['r_hw_sw_sq_r'] = constraints_vector[50:60]
    constraints_dict['log_sigma_hw_b'] = constraints_vector[60:70]
    constraints_dict['r_hw_sw_sq_b'] = constraints_vector[70:80]

    return constraints_dict


def get_best_fit_param_dict(path):

    parameters = Table.read(path + 'parameters.hdf5', path='parameters')

    param_dict = {}

    live_points = np.genfromtxt(path + 'phys_live.points')
    best_fit = live_points[np.argmax(live_points[:, -2])]

    sel = np.cumsum(parameters['fit']) - 1

    for i in range(len(parameters)):

        if parameters['fit'][i]:
            param_dict[parameters['name'][i]] = best_fit[sel[i]]
        else:
            param_dict[parameters['name'][i]] = parameters['default'][i]

    return param_dict


def get_param_dict_sample(path):

    parameters = Table.read(path + 'parameters.hdf5', path='parameters')

    sample = []
    data = np.genfromtxt(path + 'post_equal_weights.dat')[:, :-1]

    for i in range(len(data)):

        param_dict = {}

        sel = np.cumsum(parameters['fit']) - 1

        for k in range(len(parameters)):

            if parameters['fit'][k]:
                param_dict[parameters['name'][k]] = data[i, :][sel[k]]
            else:
                param_dict[parameters['name'][k]] = parameters['default'][k]

        sample.append(param_dict)

    return sample


def surface_density_from_catalog(catalog, log_lum_bins, rp_bins,
                                 sigma_200_max=3.0, interlopers=True):

    pairs = pairs_from_catalog(catalog)

    sigma = np.zeros((len(log_lum_bins) - 1, len(rp_bins) - 1))

    for i in range(len(log_lum_bins) - 1):
        mask = ((log_lum_bins[i] < np.log10(pairs['lum_prim'])) &
                (np.log10(pairs['lum_prim']) < log_lum_bins[i + 1]) &
                (np.abs(pairs['vz']) < sigma_200_max * 200 *
                 sigma_200(pairs['lum_prim'], pairs['color_prim'])))

        if not interlopers:
            mask = mask & (~pairs['interloper'])

        weights = (pairs['weight_spec_prim'][mask] *
                   pairs['weight_spec_secd'][mask])

        sigma[i, :], rp_bins = np.histogram(pairs['rp'][mask], rp_bins,
                                            weights=weights)
        sigma[i] = sigma[i] / (np.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2))
        sigma[i] = sigma[i] / np.sum(pairs['weight_spec_prim'][mask]
                                     [pairs['prim_pair'][mask]])

    return sigma


def remove_bnc(catalog):

    centrals = catalog[catalog['gal_type'] == 'centrals']

    idx1, idx2 = crossmatch(catalog['halo_id'], centrals['halo_id'])
    lum_cen = np.copy(catalog['luminosity'])
    lum_cen[idx1] = centrals[idx2]['luminosity']

    return catalog[(catalog['luminosity'] < lum_cen) |
                   (catalog['gal_type'] == 'centrals')]


def count_neighbours(ra1, dec1, ra2, dec2, alpha=1.5):
    """
    Finds matches in one catalog to another.

    Parameters
    ra1 : array-like
        Right Ascension in degrees of the first catalog
    dec1 : array-like
        Declination in degrees of the first catalog (shape of array must match
        `ra1`)
    ra2 : array-like
        Right Ascension in degrees of the second catalog
    dec2 : array-like
        Declination in degrees of the second catalog (shape of array must match
        `ra2`)
    alpha : float, optional
        The radius in which neighbours are counted.

    Returns
    -------
    n : int array
        Indecies into the number of neighbours in the second catalog. Will
        never be larger than `ra1`/`dec1`.
    """

    ra1 = np.array(ra1, copy=False)
    dec1 = np.array(dec1, copy=False)
    ra2 = np.array(ra2, copy=False)
    dec2 = np.array(dec2, copy=False)

    if ra1.shape != dec1.shape:
        raise ValueError('ra1 and dec1 do not match!')
    if ra2.shape != dec2.shape:
        raise ValueError('ra2 and dec2 do not match!')

    x1, y1, z1 = _spherical_to_cartesian(ra1.ravel(), dec1.ravel())

    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords1 = np.empty((x1.size, 3))
    coords1[:, 0] = x1
    coords1[:, 1] = y1
    coords1[:, 2] = z1

    x2, y2, z2 = _spherical_to_cartesian(ra2.ravel(), dec2.ravel())

    # this is equivalent to, but faster than just doing np.array([x1, y1, z1])
    coords2 = np.empty((x2.size, 3))
    coords2[:, 0] = x2
    coords2[:, 1] = y2
    coords2[:, 2] = z2

    kdt1 = KDT(coords1)
    kdt2 = KDT(coords2)

    r = alpha * 2 * np.pi / 360.0
    if alpha > 0.1:
        r = np.sqrt(2 - 2 * np.cos(r))

    results = kdt1.query_ball_tree(kdt2, r=r)

    return np.array([(len(results[i]) - 1) for i in range(len(results))])


def fiber_collisions(ra, dec, z, f_col=0.35, alpha=1.5, f_dec=0.99, seed=None):

    z_spec = np.copy(z)

    status = np.zeros(len(z), dtype=np.int16)
    # 0: no assignment yet
    # 1: decollided and with redshift
    # 2: (potentially) collided or without redshift due to other problem

    r_coll = 55.0 / 3600.0  # in degrees
    k_max = 6
    idx = np.zeros([k_max, len(ra)], dtype=np.int32)
    dr = np.zeros([k_max, len(ra)], dtype=np.float32)
    for k in range(k_max):
        # Note that nnearest must be at least 2 because the nearest "neighbour"
        # is the galaxy itself.
        dummy, idx[k], dr[k] = spherematch(ra, dec, ra, dec, nnearest=k + 2)

    # targets without neighbours are always decollided
    near = dr < r_coll
    n_coll = np.sum(near, axis=0)
    status[n_coll == 0] = 1

    with NumpyRNGContext(seed):
        for k in range(1, k_max + 1):
            mask = (n_coll == k) & (status == 0)
            sel = np.arange(len(mask))[mask]
            np.random.shuffle(sel)
            for i in sel:
                # only continues if no assignment so far
                if status[i] == 0:
                    # adds that galaxy to the decollided sample
                    status[i] = 1
                    # adds all close neighbours to the potentially collided
                    # sample
                    for j in range(k):
                        status[idx[j][i]] = 2

        # determines collided galaxies from the potentially collided ones
        mask = status == 2

        if type(f_col) is tuple:
            n_alpha = count_neighbours(ra[mask], dec[mask], ra, dec, alpha=1.5)
            argsort = np.argsort(n_alpha)
            p_alpha = np.zeros(len(n_alpha))
            p_alpha[argsort] = (np.arange(len(n_alpha)) + 0.5) / len(n_alpha)

            p_col = f_col[0] + (f_col[1] - f_col[0]) * p_alpha
        else:
            p_col = f_col

        status[mask] = np.where(np.random.random(np.sum(mask)) < p_col, 1, 2)
        status[np.random.random(len(ra)) > f_dec] = 2

    coll = status == 2
    z_spec[coll] = z[idx[0]][coll]
    return coll, z_spec
