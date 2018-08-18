from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.integrate import quad as quad_integration

from halotools.empirical_models.phase_space_models.analytic_models.satellites.nfw.kernels.mass_profile import _g_integral
from halotools.empirical_models import BiasedNFWPhaseSpace, NFWPhaseSpace
from halotools.empirical_models import MonteCarloGalProf


def _jeans_integrand_term1(y, *args):
    r""" First term in the Jeans integrand
    """
    bias_ratio = args[0]  # = halo_conc/gal_conc
    gamma = args[1]
    return (np.log(1 + bias_ratio * y) /
            (y**(gamma + 2.0) * (1 + y)**(3.0 - gamma)))


def _jeans_integrand_term2(y, *args):
    r""" Second term in the Jeans integrand
    """
    bias_ratio = args[0]  # = halo_conc/gal_conc
    gamma = args[1]

    numerator = bias_ratio
    denominator = ((y**(gamma + 1)) * ((1 + y)**(3 - gamma)) *
                   (1 + bias_ratio * y))
    return numerator / denominator


def dimensionless_radial_velocity_dispersion(scaled_radius, halo_conc,
                                             gal_conc, gamma,
                                             profile_integration_tol=1e-4):

    x = np.atleast_1d(scaled_radius).astype(np.float64)
    x = np.where(x == 0, 1e-3, x)
    result = np.zeros_like(x)

    prefactor = (gal_conc * (gal_conc*x)**(gamma) *
                 (1. + gal_conc * x)**(3.0 - gamma) / _g_integral(halo_conc))
    extra_args = (halo_conc / gal_conc, gamma)

    lower_limit = gal_conc*x
    upper_limit = gal_conc*3
    for i in range(len(x)):
        term1, _ = quad_integration(
            _jeans_integrand_term1, lower_limit[i], upper_limit,
            epsrel=profile_integration_tol, args=extra_args, limit=5000,
            epsabs=0)
        term2, _ = quad_integration(
            _jeans_integrand_term2, lower_limit[i], upper_limit,
            epsrel=profile_integration_tol, args=extra_args, limit=5000,
            epsabs=0)
        result[i] = term1 - term2

    return np.sqrt(result * prefactor)


def cum_gal_integral_gamma_0p0(scaled_radius, gal_conc):
    return ((4 * scaled_radius * gal_conc +
             2 * (scaled_radius * gal_conc + 1.0)**2.0 *
             np.log(scaled_radius * gal_conc + 1.0) + 3.0) /
            (2 * gal_conc**3.0 * (scaled_radius * gal_conc + 1.0)**2.0))


class BiasedGeneralizedNFWPhaseSpace(BiasedNFWPhaseSpace):

    def __init__(self, profile_integration_tol=1e-5, gamma=1.0, **kwargs):

        BiasedNFWPhaseSpace.__init__(self, **kwargs)
        self.gamma = gamma

    def cumulative_gal_PDF(self, scaled_radius, halo_conc, conc_gal_bias):
        r""" Analogous to `cumulative_mass_PDF`, but for the satellite galaxy
        distribution instead of the host halo mass distribution.

        Parameters
        -------------
        scaled_radius : array_like
            Halo-centric distance *r* scaled by the halo boundary
            :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`. Can be a scalar or
            numpy array.

        halo_conc : array_like
            Value of the halo concentration. Can either be a scalar, or a numpy
            array of the same dimension as the input ``scaled_radius``.

        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.

        Returns
        -------------
        p: array_like
            The fraction of the total mass enclosed
            within the input ``scaled_radius``, in :math:`M_{\odot}/h`;
            has the same dimensions as the input ``scaled_radius``.

        Examples
        --------
        >>> model = GeneralizedBiasedNFWPhaseSpace(gamma=0.0)

        >>> scaled_radius = 0.5  # units of Rvir
        >>> halo_conc = 5
        >>> conc_gal_bias = 3
        >>> result1 = model.cumulative_gal_PDF(scaled_radius, halo_conc,
                                               conc_gal_bias)

        >>> num_halos = 50
        >>> scaled_radius = np.logspace(-2, 0, num_halos)
        >>> halo_conc = np.linspace(1, 25, num_halos)
        >>> conc_gal_bias = np.zeros(num_halos) + 2.
        >>> result2 = model.cumulative_gal_PDF(scaled_radius, halo_conc,
                                               conc_gal_bias)
        """
        gal_conc = self._clipped_galaxy_concentration(halo_conc, conc_gal_bias)

        if self.gamma == 1.0:
            return NFWPhaseSpace.cumulative_mass_PDF(self, scaled_radius,
                                                     gal_conc)
        elif self.gamma == 0.0:
            return ((cum_gal_integral_gamma_0p0(scaled_radius, gal_conc) -
                     cum_gal_integral_gamma_0p0(0.0, gal_conc)) /
                    (cum_gal_integral_gamma_0p0(1.0, gal_conc) -
                     cum_gal_integral_gamma_0p0(0.0, gal_conc)))
        else:
            print("The function ``cumulative_gal_PDF'' of the "
                  "BiasedGeneralizedNFWPhaseSpace class only supports values "
                  "of 0.0 and 1.0 for gamma.")
            raise ValueError

    def dimensionless_radial_velocity_dispersion(self, scaled_radius, halo_conc, conc_gal_bias):
        r"""
        Analytical solution to the isotropic jeans equation for an NFW potential,
        rendered dimensionless via scaling by the virial velocity.

        :math:`\tilde{\sigma}^{2}_{r}(\tilde{r})\equiv\sigma^{2}_{r}(\tilde{r})/V_{\rm vir}^{2} = \frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}`

        See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

        Parameters
        -----------
        scaled_radius : array_like
            Length-Ngals numpy array storing the halo-centric distance
            *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
            :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`.

        halo_conc : float
            Concentration of the halo.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile scaled by the virial velocity.
            The returned result has the same dimension as the input ``scaled_radius``.
        """
        gal_conc = self._clipped_galaxy_concentration(halo_conc, conc_gal_bias)
        return dimensionless_radial_velocity_dispersion(
            scaled_radius, halo_conc, gal_conc, self.gamma,
            profile_integration_tol=self._profile_integration_tol)

    def radial_velocity_dispersion(self, radius, total_mass, halo_conc, conc_gal_bias):
        r"""
        Method returns the radial velocity dispersion scaled by
        the virial velocity as a function of the halo-centric distance.

        Parameters
        ----------
        radius : array_like
            Radius of the halo in Mpc/h units; can be a float or
            ndarray of shape (num_radii, )

        total_mass : array_like
            Float or ndarray of shape (num_radii, ) storing the host halo mass

        halo_conc : array_like
            Float or ndarray of shape (num_radii, ) storing the host halo concentration

        conc_gal_bias : array_like
            Ratio of the galaxy concentration to the halo concentration,
            :math:`c_{\rm gal}/c_{\rm halo}`.

        Returns
        -------
        result : array_like
            Radial velocity dispersion profile as a function of the input ``radius``,
            in units of km/s.

        """
        virial_velocities = self.virial_velocity(total_mass)
        halo_radius = self.halo_mass_to_halo_radius(total_mass)
        scaled_radius = radius/halo_radius

        dimensionless_velocities = (
            self.dimensionless_radial_velocity_dispersion(
                scaled_radius, halo_conc, conc_gal_bias, self.gamma)
            )
        return dimensionless_velocities*virial_velocities


    def build_lookup_tables(self):
        r""" Method used to create a lookup table of the spatial and velocity radial profiles.

        Parameters
        ----------
        logrmin : float, optional
            Minimum radius used to build the spline table.
            Default is set in `~halotools.empirical_models.model_defaults`.

        logrmax : float, optional
            Maximum radius used to build the spline table
            Default is set in `~halotools.empirical_models.model_defaults`.

        Npts_radius_table : int, optional
            Number of control points used in the spline.
            Default is set in `~halotools.empirical_models.model_defaults`.

        """
        MonteCarloGalProf.build_lookup_tables(self, Npts_radius_table=1000)
