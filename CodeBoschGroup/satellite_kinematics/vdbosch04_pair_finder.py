"""
Module containing the `~halotools.mock_observables.vdbosch04_pair_finder`
function.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .vdbosch04_pair_finder_engine import vdbosch04_pair_finder_engine

from halotools.mock_observables.mock_observables_helpers import enforce_sample_respects_pbcs, get_period
from halotools.mock_observables.pair_counters.rectangular_mesh import RectangularMesh

__all__ = ('vdbosch04_pair_finder', )

__author__ = ['Johannes Ulf Lange']


def vdbosch04_pair_finder(sample, r_h, z_h, r_s, z_s, marks, ell=1, period=None,
                          approx_cell_size=None, mark_min=0):
    """
    Function for finding pairs according to vdBosch et al. 2004.
    
    Points are considered primaries if
        1. they contain no point with a higher mark within a cylindrical volume
        described by r_h and z_h and
        2. they are not contained within the cylindrical volume given by r_h and
        z_h of another point of higher mark that is considered a central.

    Points are considered secondaries if they lie within a cylindrical volume of
    a point that is considered a primary. The volume is defined by r_s and z_s
    of the central. Primaries and their secondaries form a pair.
    
    Parameters
    ----------
    sample : array_like
        Npts x 3 numpy array containing 3-D positions of points.

        See the :ref:`mock_obs_pos_formatting` documentation page, or the
        examples section below, for instructions on how to transform your
        coordinate position arrays into the format accepted by the ``sample``
        argument.
    
    r_h : array_like
        Radius of the cylinder to search for neighbors around points. If a
        single float is given, ``r_h`` is assumed to be the same for each point
        in the ``sample``.
    
    z_h : array_like
        Half the length of cylinders to search for neighbors around points. If a
        single float is given, ``z_h`` is assumed to be the same for each point
        in the ``sample``.
    
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

    ell : float
        Float describing the ellipticity of the cylinder. Instead of using a
        circle in the x-y plane to determine the distance, we use an ellipse.
        Particularly, x^2 / ellipticity^2 + y^2 < r. This is useful if
        x and y are ra and dec, i.e. angular coordinates. For ell = 1, this
        corresponds to the normal circle. If a single float is given,
        ``ell`` is assumed to be the same for each point in the ``sample``.
    
    period : array_like, optional
        Length-3 sequence defining the periodic boundary conditions in each
        dimension. If you instead provide a single scalar, Lbox, period is
        assumed to be the same in all Cartesian directions.
    
    approx_cell_size : array_like, optional
        Length-3 array serving as a guess for the optimal manner by how points
        will be apportioned into subvolumes of the simulation box.
        The optimum choice unavoidably depends on the specs of your machine.
        Default choice is to use 1.1 max(``r_h``) in the xy-dimensions
        and 1.1 max(``z_h``) in the z-dimension, which will return reasonable
        result performance for most use-cases. However, it can never be below
        1/500 of the box size in any dimension.
        Performance can vary with this parameter, so it might be beneficial
        that you experiment with this parameter when carrying out
        performance-critical calculations.

    mark_min : float
        Minimum mark for a point to be considered a primary. Using this can
        considerably speed up computations.
    
    Returns
    -------
    prim : numpy.array
        Int array containing the indices of primaries in primary-secondary
        pairs.
    
    secd : numpy.array
        Int array containing the indices of secondaries in primary-secondary
        pairs.
    
    Examples
    --------
    In this first example, we want to use the pair finder to analyze satellite
    kinematics similar to the work in More et al. 2008
    (https://arxiv.org/abs/0807.4532). We will first generate a galaxy catalog
    to analyze.
    
    >>> from halotools.empirical_models import PrebuiltSubhaloModelFactory
    >>> from halotools.sim_manager import CachedHaloCatalog
    >>> model = PrebuiltSubhaloModelFactory('behroozi10')
    >>> halocat = CachedHaloCatalog(simname='bolplanck', redshift=0)
    >>> model.populate_mock(halocat)

    We place our points into redshift-space, formatting the result into the
    appropriately shaped array used throughout the `~halotools.mock_observables`
    sub-package:
    
    >>> from halotools.mock_observables import return_xyz_formatted_array
    
    >>> mask = model.mock.galaxy_table['stellar_mass'] > 10**8.5
    >>> mstar = model.mock.galaxy_table['stellar_mass'][mask]
    >>> x = model.mock.galaxy_table['x'][mask]
    >>> y = model.mock.galaxy_table['y'][mask]
    >>> z = model.mock.galaxy_table['z'][mask]
    >>> vz = model.mock.galaxy_table['vz'][mask]
    >>> pos = return_xyz_formatted_array(x, y, z, period=halocat.Lbox,
            velocity=vz, velocity_distortion_dimension='z')
    
    We now define how r_h, z_h, r_s and z_s depend on the properties of each
    galaxy. Similar to More et al. 2008, we make the parameters depend on the
    average velocity dispersion as a function of stellar mass. We use the
    fit of \sigma_200 to M_\star from More et al. 2010. In this works z_h is
    defined in units of km/s. To convert that into Mpc/h, we need to divide by
    100 since all units in halotools assume *h=1*.
    
    >>> sigma_200 = (10**(2.07 + 0.22 * (np.log10(mstar) - 10) + 0.21 *
    >>>                   (np.log10(mstar) - 10)**2) / 200)
    >>> r_h = 0.8 * sigma_200
    >>> z_h = 2000 * sigma_200 / 100.0
    >>> r_s = 0.15 * sigma_200
    >>> z_s = 1000 * sigma_200 / 100.0

    ...
    """
    
    # Processes the inputs with the helper function.
    args = _vdbosch04_pair_finder_process_args(
        sample, r_h, z_h, r_s, z_s, marks, period, approx_cell_size, ell,
        mark_min)
    
    x_in, y_in, z_in, r_h_in, z_h_in, r_s_in, z_s_in = args[0:7]
    period, pbcs_in, approx_cell_size, marks, ell, mark_min = args[7:13]
    x_period, y_period, z_period = period
    approx_x_cell_size, approx_y_cell_size, approx_z_cell_size = \
        approx_cell_size
    
    # Builds the rectangular mesh.
    mesh = RectangularMesh(x_in, y_in, z_in, x_period, y_period, z_period,
                           approx_x_cell_size, approx_y_cell_size,
                           approx_z_cell_size)
    
    # Run the engine.
    prim, secd = vdbosch04_pair_finder_engine(
        mesh, x_in, y_in, z_in, marks, r_h_in, z_h_in, r_s_in, z_s_in, pbcs_in,
        ell, mark_min)
    
    return prim, secd


def _get_distance(sample, distance):
    """ This helper function processes the input ``distance`` value and returns
    the appropriate array after requiring the input is the appropriate
    size and verifying that all entries are bounded positive numbers.
    """
    npts = len(sample)
    distance = np.atleast_1d(distance).astype(float)

    if len(distance) == 1:
        distance = np.array([distance[0]]*npts)
    else:
        try:
            assert len(distance) == npts
        except AssertionError:
            msg = "Input r_h, z_h, r_s, z_s and ell must be the same length " \
                  "as the sample."
            raise ValueError(msg)

    try:
        assert np.all(distance < np.inf)
        assert np.all(distance > 0)
    except AssertionError:
        msg = "Input r_h, z_h, r_s, z_s and ell must be an array of bounded " \
              "positive numbers."
        raise ValueError(msg)

    return np.array(distance)


def _enclose_in_box(x, y, z, min_size=None):
    """
    Builds a box which encloses all points, shifting the points so that
    the "leftmost" point is (0,0,0).

    Parameters
    ----------
    x,y,z : array_like
        Cartesian positions of points.

    min_size : array_like
        Minimum lengths of a side of the box.  If the minimum box constructed
        around the points has a side i less than ``min_size[i]``, then the box
        is padded in order to obtain the minimum specified size.

    Returns
    -------
    x, y, z, Lbox
        shifted positions and box size.
    """
    xmin = np.min(x)
    ymin = np.min(y)
    zmin = np.min(z)
    xmax = np.max(x)
    ymax = np.max(y)
    zmax = np.max(z)

    xyzmax = np.array([xmax, ymax, zmax]) - np.array([xmin, ymin, zmin])

    x -= xmin
    y -= ymin
    z -= zmin

    Lbox = xyzmax

    if min_size is not None:
        if np.any(Lbox < min_size):
            Lbox[(Lbox < min_size)] = min_size[(Lbox < min_size)]

    return x, y, z, Lbox


def _set_approximate_cell_size(approx_cell_size, r_max, z_max, period):
    """
    Processes the approximate cell size parameters.
    If either is set to None, apply default settings.
    """

    # Set the approximate cell sizes of the trees

    if approx_cell_size is None:
        approx_cell_size = np.maximum(np.array([r_max, r_max, z_max]) *
                                      1.1, period / 500.0)
    else:
        try:
            assert len(approx_cell_size) == 3
            assert type(approx_cell_size) is np.ndarray
            assert approx_cell_size.ndim == 1
        except AssertionError:
            msg = "Input ``approx_cell_size`` must be a length-3 array."
            raise ValueError(msg)
        
        try:
            assert np.all(period / approx_cell_size < 500.0)
        except AssertionError:
            msg = ("Input ``approx_cell_size`` cannot be smaller than "
                   "``period`` / 500 in any dimension.")
            raise ValueError(msg)

    return approx_cell_size


def _vdbosch04_pair_finder_process_args(sample, r_h, z_h, r_s, z_s, marks,
                                        period, approx_cell_size, ell,
                                        mark_min):
    """
    Private function to process the arguments for vdbosch04_pair_finder.
    """

    r_h = _get_distance(sample, r_h)
    z_h = _get_distance(sample, z_h)
    r_s = _get_distance(sample, r_s)
    z_s = _get_distance(sample, z_s)
    ell = _get_distance(sample, ell)
    r_max = np.amax(np.maximum(r_h, r_s))
    z_max = np.amax(np.maximum(z_h, z_s))

    period, PBCs = get_period(period)

    # At this point, period may still be set to None, in which case we must
    # remap our points inside the smallest enclosing cube and set ``period``
    # equal to this cube size.
    if period is None:
        x, y, z, period = (
            _enclose_in_box(
                sample[:, 0], sample[:, 1], sample[:, 2],
                min_size=np.array([max_r_h*3.0, max_r_h*3.0, max_z_h*3.0])))
    else:
        x = sample[:, 0]
        y = sample[:, 1]
        z = sample[:, 2]

    enforce_sample_respects_pbcs(x, y, z, period)
    
    approx_cell_size = _set_approximate_cell_size(
        approx_cell_size, r_max, z_max, period)
    
    try:
        assert len(marks) == len(sample)
    except AssertionError:
        msg = "``marks`` and ``sample`` must have the same length."
        raise ValueError(msg)
    
    return (x, y, z, r_h, z_h, r_s, z_s, period, PBCs, approx_cell_size, marks,
            ell, mark_min)
