import numpy as np
from astropy.utils.misc import NumpyRNGContext

from halotools.custom_exceptions import HalotoolsError
from halotools.utils.array_utils import custom_len
from halotools.empirical_models import Cacciato09Cens, Cacciato09Sats
from halotools.empirical_models.occupation_models import OccupationComponent

__all__ = ('Lange18Cens', 'Lange18Sats')


class Lange18CensNoColor(Cacciato09Cens):

    def get_published_parameters(self):
        r""" Return the values of ``self.param_dict`` according to
        the best-fit values of the WMAP3 model in Table 3 of arXiv:0807.4932.
        In this analysis, halo masses have been defined using an overdensity of
        180 times the background density of the Universe.
        """
        param_dict = (
            {'log_L_0': 9.935,
             'log_M_1': 11.07,
             'gamma_1': 3.273,
             'gamma_2': 0.255,
             'gamma_3': 0.000,
             'sigma': 0.143}
            )

        return param_dict

    def median_prim_galprop(self, **kwargs):
        r""" Return the median primary galaxy property of a central galaxy as a
        function of the input table.
        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.
        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.
        Returns
        -------
        prim_galprop : array_like
            Array containing the median primary galaxy property of the halos
            specified.
        """

        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``median_prim_galprop`` function of the "
                   "``Cacciato09Cens`` class.\n")
            raise HalotoolsError(msg)

        gamma_1 = self.param_dict['gamma_1']
        gamma_2 = self.param_dict['gamma_2']
        gamma_3 = self.param_dict['gamma_3']
        mass_c = 10**self.param_dict['log_M_1']
        prim_galprop_c = 10**self.param_dict['log_L_0']

        r = mass / mass_c

        return (prim_galprop_c * (r / (1 + r))**gamma_1 * (1 + r)**gamma_2 *
                r**(gamma_3 * np.log10(r)))


class Lange18Cens(OccupationComponent):
    """ CLF-style model for the central galaxy occupation. Since it is a CLF
    model, it also assigns luminosities to galaxies in addition to effectively
    being an HOD model.

    """

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_mvir',
                 prim_galprop_key='luminosity', color_key='color', **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample in h=1 solar
            luminosity units.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property
            governing the occupation statistics of central galaxies.

        prim_galprop_key : string, optional
            String giving the column name of the primary galaxy property that
            is assigned.

        color_key : string, optional
            String giving the column name of the galaxy color that is assigned.

        Examples
        --------
        >>> cen_model = More11Cens()
        >>> cen_model = More11Cens(threshold = 11.25)
        >>> cen_model = More11Cens(prim_haloprop_key = 'halo_mvir')

        """

        super(Lange18Cens, self).__init__(
            gal_type='centrals', threshold=threshold,
            upper_occupation_bound=1.0,
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self._mock_generation_calling_sequence = ['mc_occupation', 'mc_color',
                                                  'mc_prim_galprop']
        self.prim_galprop_key = prim_galprop_key
        self.color_key = color_key
        self._galprop_dtypes_to_allocate = np.dtype([(prim_galprop_key, 'f8'),
                                                     (color_key, object)])
        self.param_dict = self.get_published_parameters()
        self._methods_to_inherit = (['mc_occupation',
                                     'median_prim_galprop_red',
                                     'median_prim_galprop_blue',
                                     'mean_occupation_red',
                                     'mean_occupation_blue',
                                     'mean_occupation',
                                     'mc_prim_galprop',
                                     'clf_red', 'clf_blue', 'clf',
                                     'mean_red_fraction'])

        self.cens_red = Lange18CensNoColor(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key)
        self.cens_blue = Lange18CensNoColor(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key)

    def get_published_parameters(self):

        param_dict = (
            {'log_L_0_r': 9.99,
             'log_M_1_r': 11.50,
             'gamma_1_r': 4.88,
             'gamma_2_r': 0.31,
             'gamma_3_r': 0.00,
             'sigma_r': 0.20,
             'log_L_0_b': 9.55,
             'log_M_1_b': 10.55,
             'gamma_1_b': 2.13,
             'gamma_2_b': 0.34,
             'gamma_3_b': 0.00,
             'sigma_b': 0.24,
             'f_0': 0.70,
             'alpha_f': 0.15,
             'beta_f': 0.00}
        )

        return param_dict

    def _update_param_dict(self):

        for key in self.cens_red.param_dict.keys():
            self.cens_red.param_dict[key] = self.param_dict[key + '_r']

        for key in self.cens_blue.param_dict.keys():
            self.cens_blue.param_dict[key] = self.param_dict[key + '_b']

    def mean_red_fraction(self, **kwargs):
        """
        """
        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``mean_red_fraction`` function of the "
                   "``More11Cens`` class.\n")
            raise HalotoolsError(msg)

        red_fraction = (self.param_dict['f_0'] + self.param_dict['alpha_f'] *
                        (np.log10(mass) - 12.0) + self.param_dict['beta_f'] *
                        (np.log10(mass) - 12.0)**2)
        if np.abs(self.param_dict['beta_f']) > 1e-9:
            stat_point = - self.param_dict['alpha_f'] / (
                2.0 * self.param_dict['beta_f'])
            stat_value = (self.param_dict['f_0'] -
                          self.param_dict['alpha_f']**2 /
                          (4.0 * self.param_dict['beta_f']))
            if self.param_dict['beta_f'] < 0:
                red_fraction = np.where(np.log10(mass) - 12.0 > stat_point,
                                        stat_value, red_fraction)
            else:
                red_fraction = np.where(np.log10(mass) - 12.0 < stat_point,
                                        stat_value, red_fraction)
        return np.minimum(np.maximum(red_fraction, 0), 1)

    def mc_color(self, **kwargs):
        """
        """

        self._update_param_dict()

        mean_occ_r = self.mean_occupation_red(**kwargs)
        mean_occ = self.mean_occupation(**kwargs)
        red_fraction = self.mean_red_fraction(**kwargs)

        red_fraction = mean_occ_r * red_fraction / mean_occ

        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(red_fraction))

        result = np.where(mc_generator < red_fraction, 'red', 'blue')
        if 'table' in kwargs:
            kwargs['table'][self.color_key][:] = result

        return result

    def median_prim_galprop_red(self, **kwargs):
        """ Return the median primary galaxy property of a central galaxy as a
        function of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        prim_galprop : array_like
            Array containing the median primary galaxy property of the halos
            specified.
        """

        self._update_param_dict()

        return self.cens_red.median_prim_galprop(**kwargs)

    def median_prim_galprop_blue(self, **kwargs):
        """ Return the median primary galaxy property of a central galaxy as a
        function of the input table.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        prim_galprop : array_like
            Array containing the median primary galaxy property of the halos
            specified.
        """

        self._update_param_dict()

        return self.cens_blue.median_prim_galprop(**kwargs)

    def clf_red(self, prim_galprop=1e10, prim_haloprop=1e12):
        """ Return the CLF in units of dn/dlogL for the primary halo property
        and galaxy property L.

        Parameters
        ----------
        prim_haloprop : array_like, optional
            Array of mass-like variable upon which the calculation is based.

        prim_galprop : array_like, optional
            Array of luminosity-like variable of the galaxy upon which the
            calculation is based.

        Returns
        -------
        clf : array_like
            Array containing the CLF in units of dN/dlogL. If ``prim_haloprop``
            has only one element or is a scalar, the same primary halo property
            is assumed for all CLF values. Similarly, if ``prim_galprop`` has
            only one element or is a scalar, the same primary galaxy property
            is assumed throughout.
        """

        self._update_param_dict()

        return self.cens_red.clf(prim_galprop=prim_galprop,
                                 prim_haloprop=prim_haloprop)

    def clf_blue(self, prim_galprop=1e10, prim_haloprop=1e12):
        """ Return the CLF in units of dn/dlogL for the primary halo property
        and galaxy property L.

        Parameters
        ----------
        prim_haloprop : array_like, optional
            Array of mass-like variable upon which the calculation is based.

        prim_galprop : array_like, optional
            Array of luminosity-like variable of the galaxy upon which the
            calculation is based.

        Returns
        -------
        clf : array_like
            Array containing the CLF in units of dN/dlogL. If ``prim_haloprop``
            has only one element or is a scalar, the same primary halo property
            is assumed for all CLF values. Similarly, if ``prim_galprop`` has
            only one element or is a scalar, the same primary galaxy property
            is assumed throughout.
        """

        self._update_param_dict()

        return self.cens_blue.clf(prim_galprop=prim_galprop,
                                  prim_haloprop=prim_haloprop)

    def clf(self, prim_galprop=1e10, prim_haloprop=1e12):
        """ Return the CLF in units of dn/dlogL for the primary halo property
        and galaxy property L.

        Parameters
        ----------
        prim_haloprop : array_like, optional
            Array of mass-like variable upon which the calculation is based.

        prim_galprop : array_like, optional
            Array of luminosity-like variable of the galaxy upon which the
            calculation is based.

        Returns
        -------
        clf : array_like
            Array containing the CLF in units of dN/dlogL. If ``prim_haloprop``
            has only one element or is a scalar, the same primary halo property
            is assumed for all CLF values. Similarly, if ``prim_galprop`` has
            only one element or is a scalar, the same primary galaxy property
            is assumed throughout.
        """

        self._update_param_dict()
        f_r = self.mean_red_fraction(prim_galprop=prim_galprop,
                                     prim_haloprop=prim_haloprop)
        phi_r = self.cens_red.clf(prim_galprop=prim_galprop,
                                  prim_haloprop=prim_haloprop)
        phi_b = self.cens_blue.clf(prim_galprop=prim_galprop,
                                   prim_haloprop=prim_haloprop)

        return f_r * phi_r + (1 - f_r) * phi_b

    def mean_occupation_red(self, prim_galprop_min=None, **kwargs):
        """ Expected number of central galaxies in a halo. Derived from
        integrating the CLF from the luminosity threshold to infinity.

        Parameters
        ----------
        prim_galprop_min : float, optional
            Lower limit of the CLF integration used to calculate the expected
            number of satellite galaxies. If not specified, the lower limit is
            the threshold.

        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are
            based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the halo of the input mass.
        """

        self._update_param_dict()

        return self.cens_red.mean_occupation(prim_galprop_min=prim_galprop_min,
                                             **kwargs)

    def mean_occupation_blue(self, prim_galprop_min=None, **kwargs):
        """ Expected number of central galaxies in a halo. Derived from
        integrating the CLF from the luminosity threshold to infinity.

        Parameters
        ----------
        prim_galprop_min : float, optional
            Lower limit of the CLF integration used to calculate the expected
            number of satellite galaxies. If not specified, the lower limit is
            the threshold.

        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are
            based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the halo of the input mass.
        """

        self._update_param_dict()

        return self.cens_blue.mean_occupation(
            prim_galprop_min=prim_galprop_min, **kwargs)

    def mean_occupation(self, prim_galprop_min=None, **kwargs):
        """ Expected number of central galaxies in a halo. Derived from
        integrating the CLF from the luminosity threshold to infinity.

        Parameters
        ----------
        prim_galprop_min : float, optional
            Lower limit of the CLF integration used to calculate the expected
            number of satellite galaxies. If not specified, the lower limit is
            the threshold.

        prim_haloprop : array, optional
            Array of mass-like variable upon which occupation statistics are
            based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        mean_ncen : array
            Mean number of central galaxies in the halo of the input mass.
        """

        self._update_param_dict()

        return (self.mean_occupation_red(prim_galprop_min, **kwargs) *
                self.mean_red_fraction(**kwargs) +
                self.mean_occupation_blue(prim_galprop_min, **kwargs) *
                (1.0 - self.mean_red_fraction(**kwargs)))

    def mc_prim_galprop(self, **kwargs):
        """ Method to generate Monte Carlo realizations of the primary galaxy
        property of galaxies.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the primary galaxy
            properties are based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        seed : int, optional
            Random number seed used to generate the Monte Carlo realization.
            Default is None.

        Returns
        -------
        mc_prim_galprop : array
            Float array giving the Monte Carlo realization of primary galaxy
            properties of centrals in halos of the given mass.
        """

        self._update_param_dict()

        if 'table' in list(kwargs.keys()):
            color = kwargs['table'][self.color_key]
            prim_haloprop = kwargs['table'][self.prim_haloprop_key]
        elif 'color' in list(kwargs.keys()):
            color = kwargs['color']
            prim_haloprop = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``color`` "
                   "argument to the ``mc_prim_galprop`` function of the "
                   "``More11Cens`` class.\n")
            raise HalotoolsError(msg)

        color = np.atleast_1d(color)
        prim_galprop = np.zeros(len(prim_haloprop))

        seed = kwargs.get('seed', None)

        if len(color) != 1 and len(color) != len(prim_haloprop):
            msg = ("\nThe ``color`` argument must either be a scalar or "
                   "an array with the same length as the number of halos.\n")
            raise HalotoolsError(msg)

        if len(color) == 1:
            if color[0] == 'red':
                prim_galprop = self.cens_red.mc_prim_galprop(
                    prim_haloprop=prim_haloprop, seed=seed)
            else:
                prim_galprop = self.cens_blue.mc_prim_galprop(
                    prim_haloprop=prim_haloprop, seed=seed)
        else:
            mask = color == 'red'
            prim_galprop[mask] = self.cens_red.mc_prim_galprop(
                prim_haloprop=prim_haloprop[mask], seed=seed)
            mask = color == 'blue'
            prim_galprop[mask] = self.cens_blue.mc_prim_galprop(
                prim_haloprop=prim_haloprop[mask], seed=seed)

        if 'table' in list(kwargs.keys()):
            kwargs['table'][self.prim_galprop_key][:] = prim_galprop

        return prim_galprop


class Lange18Sats(Cacciato09Sats):
    """ CLF-style model for the satellite galaxy occupation. Since it is a CLF
    model, it also assigns luminosities to galaxies in addition to effectively
    being an HOD model.
    """

    def __init__(self, threshold=10.0, prim_haloprop_key='halo_mvir',
                 prim_galprop_key='luminosity', color_key='color', **kwargs):
        """
        Parameters
        ----------
        threshold : float, optional
            Luminosity threshold of the mock galaxy sample in h=1 solar
            luminosity units.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property
            governing the occupation statistics of satellite galaxies.

        prim_haloprop_key : string, optional
            String giving the column name of the primary galaxy property that
            is assigned.

        Examples
        --------
        >>> sat_model = More11Sats()
        >>> sat_model = More11Sats(threshold = 11.25)
        >>> sat_model = More11Sats(prim_haloprop_key = 'halo_mvir')

        """
        super(Cacciato09Sats, self).__init__(
            gal_type='satellites', threshold=threshold,
            upper_occupation_bound=float("inf"),
            prim_haloprop_key=prim_haloprop_key,
            **kwargs)

        self._mock_generation_calling_sequence = ['mc_occupation',  'mc_color',
                                                  'mc_prim_galprop']
        self.prim_galprop_key = prim_galprop_key
        self.color_key = color_key
        self._galprop_dtypes_to_allocate = np.dtype([(prim_galprop_key, 'f8'),
                                                     (color_key, object)])
        self.param_dict = self.get_default_parameters()
        self.central_occupation_model = Lange18Cens(
            threshold=threshold, prim_haloprop_key=prim_haloprop_key,
            prim_galprop_key=prim_galprop_key, **kwargs)
        self._methods_to_inherit = (
            ['mc_occupation', 'mean_occupation', 'mc_prim_galprop', 'clf',
             'phi_sat', 'alpha_sat', 'prim_galprop_cut']
            )

    def get_default_parameters(self):
        """ Return the values of ``self.param_dict`` according to
        the best-fit values of the WMAP3 model in Table 3 of arXiv:0807.4932.
        """

        param_dict = (
            {'a_1': 0.82,
             'a_2': 0.0,
             'log_M_2': 14.28,
             'b_0': -0.766,
             'b_1': 1.008,
             'b_2': -0.094,
             'f_0_sat': 0.44,
             'alpha_f_sat': 0.14,
             'delta_1': 0.0,
             'delta_2': 0.0,
             'log_L_0_r': 9.99,
             'log_M_1_r': 11.50,
             'gamma_1_r': 4.88,
             'gamma_2_r': 0.31,
             'gamma_3_r': 0.00,
             'sigma_r': 0.20}
            )

        return param_dict

    def prim_galprop_cut(self, **kwargs):
        r""" Return the cut-off primary galaxy properties of the CLF as a
        function of the input table. See equation (38) in Cacciato et al. (2009)
        for details. The cut-off primary galaxy property refers to $\L_s^\star$.

        Parameters
        ----------
        prim_haloprop : array, optional
            Array of mass-like variable upon which the calculation is based.
            If ``prim_haloprop`` is not passed, then ``table`` keyword argument
            must be passed.

        table : object, optional
            Data table storing halo catalog.
            If ``table`` is not passed, then ``prim_haloprop`` keyword argument
            must be passed.

        Returns
        -------
        prim_galprop_cut : array_like
            Array containing the cut-off primary galaxy property of the halos
            specified.
        """

        if not ('table' in list(kwargs.keys()) or 'prim_haloprop'
                in list(kwargs.keys())):
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``prim_galprop_cut`` function of the "
                   "``Cacciato09Sats`` class.\n")
            raise HalotoolsError(msg)

        self._update_central_params()

        med_prim_galprop = (
            self.central_occupation_model.median_prim_galprop_red(**kwargs))

        return med_prim_galprop * 0.562

    def mean_red_fraction(self, **kwargs):
        """
        """
        # Retrieve the array storing the mass-like variable.
        if 'table' in list(kwargs.keys()):
            mass = kwargs['table'][self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = kwargs['prim_haloprop']
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` "
                   "argument to the ``mean_red_fraction`` function of the "
                   "``More11Cens`` class.\n")
            raise HalotoolsError(msg)

        red_fraction = (
            self.param_dict['f_0_sat'] +
            self.param_dict['alpha_f_sat'] * (np.log10(mass) - 12.0))

        return np.minimum(np.maximum(red_fraction, 0), 1)

    def mc_color(self, **kwargs):
        """
        """

        red_fraction = self.mean_red_fraction(**kwargs)

        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
            mc_generator = np.random.random(custom_len(red_fraction))

        result = np.where(mc_generator < red_fraction, 'red', 'blue')
        if 'table' in kwargs:
            kwargs['table'][self.color_key][:] = result

        return result
