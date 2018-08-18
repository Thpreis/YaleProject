from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport cython 
from libc.math cimport ceil, fabs, log10, fmin, fmax, erfc, sqrt, exp, M_PI, erf, atan
from libc.math cimport pow as cpow
from libc.stdlib cimport malloc, free
from libcpp cimport bool

from astropy.table import Table

from halotools.empirical_models import NFWProfile

from satellite_kinematics import utils



__author__ = ('Johannes Ulf Lange')
__all__ = ('AnalyticModel', )

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_gamma_inc(double a, double x)

cdef extern from "gsl/gsl_spline.h":
    ctypedef struct gsl_spline:
        pass
    gsl_spline* gsl_spline_alloc(const gsl_interp_type* T, size_t size)
    int gsl_spline_init(gsl_spline* spline, const double xa[],
        const double ya[], size_t size)
    double gsl_spline_eval(const gsl_spline* spline, double x,
        gsl_interp_accel* a)

cdef extern from "gsl/gsl_interp.h":
    ctypedef struct gsl_interp_type:
        pass
    ctypedef struct gsl_interp_accel:
        pass
    gsl_interp_type* gsl_interp_linear
    gsl_interp_type* gsl_interp_cspline
    gsl_interp_accel* gsl_interp_accel_alloc()



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef class AnalyticModel(object):
    """ Analytical model to calculate various observables as a function of the
    parameters of the More11 CLF-type model.
    """
    
    # Class variables used by ``AnalyticModel``.
    cdef int n_mvir_bins, n_prim_galprop_bins, n_prim_galprop_bins_hr
    cdef int n_prim_galprop_subbins, profile
    cdef double [:] log_mvir_bins, n_halo_mvir, cvir_median, rvir_median
    cdef double [:] log_prim_galprop_bins, log_prim_galprop_bins_hr
    cdef double [:] prim_galprop_bins_hr
    cdef gsl_spline** fraction_in_aperture_interp
    cdef gsl_spline** dispersion_in_aperture_interp
    cdef gsl_interp_accel** fraction_in_aperture_accel
    cdef gsl_interp_accel** dispersion_in_aperture_accel
    cdef double logl0_r, logm1_r, gamma_1_r, gamma_2_r, sigma_r
    cdef double logl0_b, logm1_b, gamma_1_b, gamma_2_b, sigma_b
    cdef double f_0, alpha_f, beta_f, a_1, a_2, b_0, b_1, b_2, logm2
    cdef double gamma_3_r, gamma_3_b, zeta, d_sigma_r, d_sigma_b
    cdef double log_prim_galprop_min, log_prim_galprop_max, z_halo
    cdef double a_r, b_r, c_r, a_b, b_b, c_b
    cdef double threshold
    cdef public dict param_dict
    cdef object cosmology
    
    def __cinit__(self, halocat,
                  log_prim_galprop_min=np.amin(utils.default_log_lum_bins),
                  log_prim_galprop_max=np.amax(utils.default_log_lum_bins),
                  n_prim_galprop_bins=(len(utils.default_log_lum_bins) - 1),
                  mvir_min=3e10, mvir_max=np.inf, log_mvir_spacing=0.1,
                  threshold=9.5, n_prim_galprop_subbins=5, profile=1,
                  Num_ptcl_requirement=300, a_r=utils.a_r, b_r=utils.b_r,
                  c_r=utils.c_r, a_b=utils.a_b, b_b=utils.b_b, c_b=utils.c_b,
                  cosmology=utils.cosmology, z_halo=utils.z_halo):
        """
        Parameters
        ----------
        halocat : halotools.CachedHaloCatalog
            Halo catalog used to extract halo number densities and median halo
            concentrations.

        log_prim_galprop_min : float
            Logarithm of the minimum primary galaxy property used for the bins.

        log_prim_galprop_max : float
            Logarithm of the maximum primary galaxy property used for the bins.

        n_prim_galprop_bins : int
            Number of primary galaxy property bins.

        mvir_min : float, optional
            Minimum halo mass to use.

        mvir_max : float, optional
            Maximum halo mass to use.

        log_mvir_spacing : float, optional
            The analytic model converts the input halo table into a histogram
            of halo masses. This parameters sets the spacing in logarithm of the
            virial mass.

        threshold : float, optional
            Logarithm of the primary galaxy property threshold on which the
            calculations are based.

        n_prim_galprop_subbins : int, optional
            The analytic model divides every primary galaxy property bin into
            smaller subbins to make the calculations more accurate. This
            argument determines how many subbins are used.

        profile : int, optional
            Determines the radial profile of satellites.
        """
        self.param_dict = self.get_default_parameters()
        self.update_params()
        
        self.log_prim_galprop_min = log_prim_galprop_min
        self.log_prim_galprop_max = log_prim_galprop_max
        self.log_prim_galprop_bins = np.ascontiguousarray(np.linspace(
            log_prim_galprop_min, log_prim_galprop_max,
            num=(n_prim_galprop_bins + 1)))
        self.n_prim_galprop_bins = n_prim_galprop_bins
        self.log_prim_galprop_bins_hr = np.ascontiguousarray(np.linspace(
            log_prim_galprop_min, log_prim_galprop_max,
            num=(n_prim_galprop_bins * n_prim_galprop_subbins + 1)))
        self.prim_galprop_bins_hr = np.ascontiguousarray(10**np.linspace(
            log_prim_galprop_min, log_prim_galprop_max,
            num=(n_prim_galprop_bins * n_prim_galprop_subbins + 1)))
        self.n_prim_galprop_bins_hr = (n_prim_galprop_bins *
            n_prim_galprop_subbins)
        self.n_prim_galprop_subbins = n_prim_galprop_subbins
        self.threshold = threshold
        self.cosmology = cosmology
        self.z_halo = z_halo

        try:
            assert profile == 1 or profile == 2 or profile == 3
        except AssertionError:
            print("The input ``profile`` must be 1, 2 or 3.")
            raise
        self.profile = profile

        self.a_r = a_r
        self.b_r = b_r
        self.c_r = c_r
        self.a_b = a_b
        self.b_b = b_b
        self.c_b = c_b
        
        self.get_halo_properties(halocat, mvir_min, mvir_max, log_mvir_spacing,
                                 Num_ptcl_requirement=Num_ptcl_requirement)
    
    def get_default_parameters(self):
        
        param_dict = ({
            'log_L_0_r': 9.99,
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
            'beta_f': 0.00,
            'a_1': 0.82,
            'a_2': 0.0,
            'log_M_2': 14.28,
            'b_0': -0.766,
            'b_1': 1.008,
            'b_2': -0.094,
            'zeta': 1.0,
            'd_sigma_r': 0.0,
            'd_sigma_b': 0.0})
        
        return param_dict
    
    def update_params(self):
        self.logl0_r = self.param_dict['log_L_0_r']
        self.logm1_r = self.param_dict['log_M_1_r']
        self.gamma_1_r = self.param_dict['gamma_1_r']
        self.gamma_2_r = self.param_dict['gamma_2_r']
        self.gamma_3_r = self.param_dict['gamma_3_r']
        self.sigma_r = self.param_dict['sigma_r']
        self.logl0_b = self.param_dict['log_L_0_b']
        self.logm1_b = self.param_dict['log_M_1_b']
        self.gamma_1_b = self.param_dict['gamma_1_b']
        self.gamma_2_b = self.param_dict['gamma_2_b']
        self.gamma_3_b = self.param_dict['gamma_3_b']
        self.sigma_b = self.param_dict['sigma_b']
        self.f_0 = self.param_dict['f_0']
        self.alpha_f = self.param_dict['alpha_f']
        self.beta_f = self.param_dict['beta_f']
        self.a_1 = self.param_dict['a_1']
        self.a_2 = self.param_dict['a_2']
        self.b_0 = self.param_dict['b_0']
        self.b_1 = self.param_dict['b_1']
        self.b_2 = self.param_dict['b_2']
        self.logm2 = self.param_dict['log_M_2']
        self.zeta = self.param_dict['zeta']
        self.d_sigma_r = self.param_dict['d_sigma_r']
        self.d_sigma_b = self.param_dict['d_sigma_b']
    
    def get_halo_properties(self, halocat, mvir_min=0, mvir_max=np.inf,
            log_mvir_spacing=0.1, Num_ptcl_requirement=300):

        nfw = NFWProfile(cosmology=self.cosmology, redshift=self.z_halo)
        
        halos = halocat.halo_table.copy()
        mask = ((halos['halo_mvir'] >= Num_ptcl_requirement *
                 halocat.particle_mass) & (halos['halo_upid'] == -1) &
                (mvir_min < halos['halo_mvir']) &
                (halos['halo_mvir'] < mvir_max))
        halos = halos[mask]
        
        # Determine minimum and maximum of halo table bins and defines bins.
        log_mvir_min = np.min(np.log10(halos['halo_mvir']))
        log_mvir_max = np.max(np.log10(halos['halo_mvir']))
        self.n_mvir_bins = (log_mvir_max - log_mvir_min) / log_mvir_spacing
        self.log_mvir_bins = np.ascontiguousarray(np.linspace(log_mvir_min,
            log_mvir_max, num=self.n_mvir_bins + 1))

        # Analyze the properties of halos in the halocat.
        self.n_halo_mvir = np.ascontiguousarray(np.zeros(self.n_mvir_bins))
        self.cvir_median = np.ascontiguousarray(np.zeros(self.n_mvir_bins))
        self.rvir_median = np.ascontiguousarray(np.zeros(self.n_mvir_bins))
        
        for i in range(self.n_mvir_bins):
            mask = ((self.log_mvir_bins[i] < np.log10(halos['halo_mvir'])) &
                    (np.log10(halos['halo_mvir']) < self.log_mvir_bins[i+1]))
            self.n_halo_mvir[i] = float(np.sum(mask)) / np.prod(halocat.Lbox)
            self.rvir_median[i] = nfw.halo_mass_to_halo_radius(
                10**(0.5 * (self.log_mvir_bins[i] + self.log_mvir_bins[i+1])))
            self.rvir_median[i] = self.rvir_median[i] * (1 + self.z_halo)
            
            if self.n_halo_mvir[i] > 0:
                self.cvir_median[i] = np.median((halos['halo_rvir'] /
                    halos['halo_rs'])[mask])

        # Import lookup tables for the fraction of satellites in the aperture.
        self.fraction_in_aperture_interp = (<gsl_spline**> malloc((
            self.n_mvir_bins) * sizeof(gsl_spline*)))
        self.fraction_in_aperture_accel = (<gsl_interp_accel**> malloc((
            self.n_mvir_bins) * sizeof(gsl_interp_accel*)))

        cdef double r[100]
        cdef double f[100]
        
        for i in range(self.n_mvir_bins):

            self.fraction_in_aperture_interp[i] = gsl_spline_alloc(
                gsl_interp_linear, 100)
            self.fraction_in_aperture_accel[i] = gsl_interp_accel_alloc()
            
            fname = (utils.lib_directory + 'lookup/fraction_in_aperture' +
                     '_%d.hdf5' % (self.profile))
            cvir = max(np.round(self.cvir_median[i], decimals=1), 2.5)
            
            table = Table.read(fname, path='data')
            r = np.ascontiguousarray(table['r'])
            f = np.ascontiguousarray(table['c=%.1f' % cvir])

            gsl_spline_init(self.fraction_in_aperture_interp[i], r, f, 100)
        
        # Import lookup tables for the average velocity dispersion of true
        # satellites inside the aperture.
        
        self.dispersion_in_aperture_interp = (<gsl_spline**> malloc((
            self.n_mvir_bins) * sizeof(gsl_spline*)))
        self.dispersion_in_aperture_accel = (<gsl_interp_accel**> malloc((
            self.n_mvir_bins) * sizeof(gsl_interp_accel*)))
        
        cdef double vd[100]
        
        for i in range(self.n_mvir_bins):

            self.dispersion_in_aperture_interp[i] = gsl_spline_alloc(
                gsl_interp_linear, 100)
            self.dispersion_in_aperture_accel[i] = gsl_interp_accel_alloc()

            fname = (utils.lib_directory + 'lookup/velocity_dispersion' +
                     '_%d.hdf5' % (self.profile))
            cvir = max(np.round(self.cvir_median[i], decimals=1), 2.5)
            mvir = 10**(0.5 * (self.log_mvir_bins[i] + self.log_mvir_bins[i+1]))
            vvir = nfw.virial_velocity(mvir)
            
            table = Table.read(fname, path='data')
            r = np.ascontiguousarray(table['r'])
            vd = np.ascontiguousarray(table['c=%.1f' % cvir] * vvir**2)

            gsl_spline_init(self.dispersion_in_aperture_interp[i], r, vd,
                            100)

    
    cdef double log_median_prim_galprop(self, double log_prim_haloprop, int color):
        
        cdef double ratio
        
        if color == 0:
            ratio = cpow(10, log_prim_haloprop - self.logm1_r)
            return (self.logl0_r + log10(ratio / (1 + ratio)) * self.gamma_1_r
                + log10(1 + ratio) * self.gamma_2_r +
                (log_prim_haloprop - self.logm1_r) *
                (log_prim_haloprop - self.logm1_r) * self.gamma_3_r)
        
        elif color == 1:
            ratio = cpow(10, log_prim_haloprop - self.logm1_b)
            return (self.logl0_b + log10(ratio / (1 + ratio)) * self.gamma_1_b
                + log10(1 + ratio) * self.gamma_2_b +
                (log_prim_haloprop - self.logm1_b) *
                (log_prim_haloprop - self.logm1_b) * self.gamma_3_b)
    
    
    cdef double mean_red_fraction(self, double log_prim_haloprop):

        cdef double log_M12 = log_prim_haloprop - 12.0

        if fabs(self.beta_f) < 1e-9:
            return fmin(fmax(self.f_0 + self.alpha_f * log_M12, 0), 1)
        
        cdef double stat_point = - self.alpha_f / (2.0 * self.beta_f)
        cdef double stat_value = (self.f_0 -
            self.alpha_f * self.alpha_f / (4.0 * self.beta_f))
        
        if ((self.beta_f < 0 and log_M12 < stat_point) or
                (self.beta_f > 0 and log_M12 > stat_point)):
            return fmin(fmax(
                self.f_0 + self.alpha_f * log_M12 + self.beta_f * log_M12 *
                log_M12, 0), 1)
        else:
            return fmin(fmax(stat_value, 0), 1)
    
    
    cdef double gaussian_integral(self, double a, double b, double mu, double scale):
        return (0.5 * (erfc((a - mu) / (sqrt(2.0) * scale)) -
            erfc((b - mu) / (sqrt(2.0) * scale))))

    cdef double upper_gaussian_integral(self, double a, double mu, double scale):
        return 0.5 * erfc((a - mu) / (sqrt(2.0) * scale))
    
    
    cdef double log_phi_sat_star(self, double log_prim_haloprop):
        return (self.b_0 + self.b_1 * (log_prim_haloprop - 12.0) + self.b_2 *
            (log_prim_haloprop - 12.0) * (log_prim_haloprop - 12.0))
    
    
    cdef double alpha_sat(self, double log_prim_haloprop):
        return -2.0 + self.a_1 * (1.- 2.0 / M_PI * atan(
            self.a_2 * (log_prim_haloprop - self.logm2)))
    
    
    cdef double log_prim_galprop_cut(self, double log_prim_haloprop):
        return self.log_median_prim_galprop(log_prim_haloprop, 0) + log10(0.562)
    
    
    cdef double mean_n_sat(self, double phi_sat_star, double alpha_sat,
                   double prim_galprop_cut, double prim_galprop_min,
                   double prim_galprop_max):
            
        cdef double norm = 0, lower_integral = 0, upper_integral = 0

        norm = 0.5 * phi_sat_star
        lower_integral = 0
        upper_integral = 0

        if prim_galprop_min < 10 * prim_galprop_cut:
            lower_integral = - gsl_sf_gamma_inc(0.5 * (alpha_sat + 1),
                prim_galprop_min * prim_galprop_min / (prim_galprop_cut *
                prim_galprop_cut))

        if prim_galprop_max < 10 * prim_galprop_cut:
            upper_integral = - gsl_sf_gamma_inc(0.5 * (alpha_sat + 1),
                prim_galprop_max * prim_galprop_max / (prim_galprop_cut *
                prim_galprop_cut))

        return norm * (upper_integral - lower_integral)
    

    cdef double f_at_least_one(self, double lam):
        if lam > 1e-6:
            return 1.0 - exp(-lam)
        else:
            return lam
    
    
    
    cdef double cylinder_selection(self, double x):
        if x <= 1.0:
            return 1.0
        else:
            return 1.0 - sqrt(1.0 - 1.0 / (x * x))
    


    def get_constraints_dict(self, sats_as_bhgs=False, correct_nsat=False,
                             cen_voff=False):
        
        self.update_params()
        
        cdef int i, j, k
        cdef double n_halo, log_median_prim_galprop, log_prim_haloprop
        cdef double alpha_sat, phi_sat_star, prim_galprop_cut
        cdef double s_r, s_b

        cdef int color
        cdef int red = 0
        cdef int blue = 1


        
        #######################
        # Cylinder Properties #
        #######################
        
        cdef double [:] r_cyl_r = np.empty(self.n_prim_galprop_bins_hr)
        cdef double [:] r_cyl_b = np.empty(self.n_prim_galprop_bins_hr)
        
        for i in range(self.n_prim_galprop_bins_hr):
            log_median_prim_galprop = 0.5 * (self.log_prim_galprop_bins_hr[i] +
                self.log_prim_galprop_bins_hr[i+1])
            r_cyl_r[i] = 0.15 * cpow(10, self.a_r + self.b_r *
                (log_median_prim_galprop - 10.0) + self.c_r *
                cpow(log_median_prim_galprop - 10.0, 2.0)) / 200.0
            r_cyl_b[i] = 0.15 * cpow(10, self.a_b + self.b_b *
                (log_median_prim_galprop - 10.0) + self.c_b *
                cpow(log_median_prim_galprop - 10.0, 2.0)) / 200.0

        
        #######################################
        # Luminosity / Stellar Mass Functions #
        #######################################
        
        cdef double [:] nd_cen_r = np.zeros(self.n_prim_galprop_bins)
        cdef double [:] nd_cen_b = np.zeros(self.n_prim_galprop_bins)
        cdef double [:] nd_sat = np.zeros(self.n_prim_galprop_bins)
        
        cdef double f_cen_r, f_cen_b
        cdef double n_cen_r, n_cen_b, n_sat

        for i in range(self.n_mvir_bins):
            
            log_prim_haloprop = 0.5 * (self.log_mvir_bins[i] +
                self.log_mvir_bins[i+1])
            f_cen_r = self.mean_red_fraction(log_prim_haloprop)
            f_cen_b = 1.0 - f_cen_r
            s_r = fmax(0.01, self.sigma_r + self.d_sigma_r *
                       (log_prim_haloprop - 14))
            s_b = fmax(0.01, self.sigma_b + self.d_sigma_b *
                       (log_prim_haloprop - 12))
            n_halo = self.n_halo_mvir[i]
            
            # centrals
            for j in range(self.n_prim_galprop_bins):
                
                n_cen_r = self.gaussian_integral(
                    self.log_prim_galprop_bins[j],
                    self.log_prim_galprop_bins[j+1],
                    self.log_median_prim_galprop(log_prim_haloprop, red),
                    s_r) * f_cen_r
                
                n_cen_b = self.gaussian_integral(
                    self.log_prim_galprop_bins[j],
                    self.log_prim_galprop_bins[j+1],
                    self.log_median_prim_galprop(log_prim_haloprop, blue),
                    s_b) * f_cen_b
                
                nd_cen_r[j] += n_cen_r * n_halo
                nd_cen_b[j] += n_cen_b * n_halo
            
            # satellites
            alpha_sat = self.alpha_sat(log_prim_haloprop)
            phi_sat_star = cpow(10, self.log_phi_sat_star(log_prim_haloprop))
            prim_galprop_cut = cpow(10, self.log_prim_galprop_cut(
                log_prim_haloprop))
            log_median_prim_galprop = self.log_median_prim_galprop(
                log_prim_haloprop, red)
            
            for j in range(self.n_prim_galprop_bins):
                
                if cpow(10, self.log_prim_galprop_bins[j]) > 10 * prim_galprop_cut:
                    break
                
                if sats_as_bhgs:
                    n_sat = self.mean_n_sat(
                        phi_sat_star, alpha_sat, prim_galprop_cut,
                        cpow(10, self.log_prim_galprop_bins[j]),
                        cpow(10, self.log_prim_galprop_bins[j+1]))
                else:
                    n_cen_r = self.upper_gaussian_integral(
                        0.5 * (self.log_prim_galprop_bins[j] +
                               self.log_prim_galprop_bins[j + 1]),
                        self.log_median_prim_galprop(log_prim_haloprop, red),
                        s_r)
                    
                    n_cen_b = self.upper_gaussian_integral(
                        0.5 * (self.log_prim_galprop_bins[j] +
                               self.log_prim_galprop_bins[j + 1]),
                        self.log_median_prim_galprop(log_prim_haloprop, blue),
                        s_b)

                    n_sat = (self.mean_n_sat(
                        phi_sat_star, alpha_sat, prim_galprop_cut,
                        cpow(10, self.log_prim_galprop_bins[j]),
                        cpow(10, self.log_prim_galprop_bins[j+1])) *
                             (f_cen_r * n_cen_r + f_cen_b * n_cen_b *
                              self.zeta))
                
                nd_sat[j] += n_sat * n_halo
        
        
        ##########################
        # Kinematics Observables #
        ##########################
        
        cdef double [:] f_in_bin = np.empty(2)
        cdef double [:] f_sat_in_ap = np.empty(2)
        cdef double [:] f_sat_in_fb = np.empty(2)
        cdef double [:] sigma_sq = np.empty(2)
        cdef double [:] n_scd = np.empty(2)
        cdef double [:] p_one_scd = np.empty(2)
        
        cdef double r_over_rvir = 0, fb_over_rvir = 0
        cdef double mean_n_sat
        cdef double log_median_prim_galprop_r, log_median_prim_galprop_b

        cdef double n_cen, n_cen_bri = 0, n_sat_bri = 0

        cdef double [:] n_pri_zero_r = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] n_pri_zero_b = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] n_pri_r = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] n_pri_b = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] n_scd_r = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] n_scd_b = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] sigma_sw_r = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] sigma_sw_b = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] sigma_hw_r = np.zeros(self.n_prim_galprop_bins_hr)
        cdef double [:] sigma_hw_b = np.zeros(self.n_prim_galprop_bins_hr)

        for i in range(self.n_mvir_bins):

            log_prim_haloprop = 0.5 * (self.log_mvir_bins[i] +
                self.log_mvir_bins[i+1])

            f_cen_r = self.mean_red_fraction(log_prim_haloprop)
            f_cen_b = 1.0 - f_cen_r
            s_r = fmax(0.01, self.sigma_r + self.d_sigma_r *
                       (log_prim_haloprop - 14))
            s_b = fmax(0.01, self.sigma_b + self.d_sigma_b *
                       (log_prim_haloprop - 12))
            n_halo = self.n_halo_mvir[i]
            alpha_sat = self.alpha_sat(log_prim_haloprop)
            phi_sat_star = cpow(10, self.log_phi_sat_star(log_prim_haloprop))
            prim_galprop_cut = cpow(10, self.log_prim_galprop_cut(
                log_prim_haloprop))
            log_median_prim_galprop_r = self.log_median_prim_galprop(
                log_prim_haloprop, red)
            log_median_prim_galprop_b = self.log_median_prim_galprop(
                log_prim_haloprop, blue)
            mean_n_sat = self.mean_n_sat(
                phi_sat_star, alpha_sat, prim_galprop_cut,
                cpow(10, self.threshold), prim_galprop_cut * 100)
            
            for j in range(self.n_prim_galprop_bins_hr):                
                
                log_median_prim_galprop = .5 * (self.log_prim_galprop_bins_hr[j]
                    + self.log_prim_galprop_bins_hr[j+1])

                if not sats_as_bhgs and correct_nsat:
                    mean_n_sat = self.mean_n_sat(
                            phi_sat_star, alpha_sat, prim_galprop_cut,
                            cpow(10, self.threshold),
                            cpow(10, log_median_prim_galprop))
                
                # Finds fraction of halos where BCGs are red/blue galaxies with
                # that particular primary galaxy property.
                if sats_as_bhgs:
                    if j == 0:
                        n_sat = self.mean_n_sat(
                            phi_sat_star, alpha_sat, prim_galprop_cut,
                            cpow(10, self.log_prim_galprop_bins_hr[j]),
                            prim_galprop_cut * 100)
                    else:
                        n_sat = n_sat_bri
                    n_sat_bri = self.mean_n_sat(
                        phi_sat_star, alpha_sat, prim_galprop_cut,
                        cpow(10, self.log_prim_galprop_bins_hr[j+1]),
                        prim_galprop_cut * 100)
                    n_sat -= n_sat_bri
                    n_cen_bri = (self.upper_gaussian_integral(
                            self.log_prim_galprop_bins_hr[j],
                            log_median_prim_galprop_r, s_r) * f_cen_r +
                                 self.upper_gaussian_integral(
                            self.log_prim_galprop_bins_hr[j],
                            log_median_prim_galprop_b, s_b) * f_cen_b)

                for color in range(2):

                    if color == red:
                        n_cen = (self.gaussian_integral(
                            self.log_prim_galprop_bins_hr[j],
                            self.log_prim_galprop_bins_hr[j+1],
                            log_median_prim_galprop_r, s_r) * f_cen_r)
                    elif color == blue:
                        n_cen = (self.gaussian_integral(
                            self.log_prim_galprop_bins_hr[j],
                            self.log_prim_galprop_bins_hr[j+1],
                            log_median_prim_galprop_b, s_b) * f_cen_b)

                    f_in_bin[color] = n_cen

                    if sats_as_bhgs:
                        if color == red:
                            f_in_bin[color] = ((n_cen * exp(-n_sat_bri)) +
                                               (1 - n_cen_bri) * exp(-n_sat_bri)
                                               * (1.0 - exp(-n_sat)))
                        elif color == blue:
                            f_in_bin[color] = (n_cen * exp(-n_sat_bri))
                
                # Calculates the expected number and velocity dispersion of
                # true halo members.
                for color in range(2):

                    if color == red:
                        r_over_rvir = r_cyl_r[j] / self.rvir_median[i]
                    elif color == blue:
                        r_over_rvir = r_cyl_b[j] / self.rvir_median[i]
                    
                    fb_over_rvir = 0.06 / self.rvir_median[i]
                    
                    r_over_rvir = fmin(r_over_rvir, 1.0)
                    fb_over_rvir = fmin(fb_over_rvir, 1.0)

                    f_sat_in_ap[color] = gsl_spline_eval(
                        self.fraction_in_aperture_interp[i], r_over_rvir,
                        self.fraction_in_aperture_accel[i])
                    f_sat_in_fb[color] = gsl_spline_eval(
                        self.fraction_in_aperture_interp[i], fb_over_rvir,
                        self.fraction_in_aperture_accel[i])
                    sigma_sq[color] = (
                        gsl_spline_eval(
                            self.dispersion_in_aperture_interp[i],
                            r_over_rvir, self.dispersion_in_aperture_accel[i])
                        * f_sat_in_ap[color] -
                        gsl_spline_eval(
                            self.dispersion_in_aperture_interp[i],
                            fb_over_rvir, self.dispersion_in_aperture_accel[i])
                        * f_sat_in_fb[color])
                    if (f_sat_in_ap[color] - f_sat_in_fb[color]) > 0:
                        sigma_sq[color] = (sigma_sq[color] /
                            (f_sat_in_ap[color] - f_sat_in_fb[color]))
                    
                    f_sat_in_ap[color] = f_sat_in_ap[color] - f_sat_in_fb[color]
                    
                    sigma_sq[color] = fmax(sigma_sq[color], 0)
                    if cen_voff:
                        sigma_sq[color] += 25 * cpow(cpow(10, log_prim_haloprop) / 0.7 / 1e11, 0.86)
                    f_sat_in_ap[color] = fmax(f_sat_in_ap[color], 0)
                    
                    if color == red:
                        n_scd[color] = mean_n_sat * f_sat_in_ap[color]
                    else:
                        n_scd[color] = mean_n_sat * f_sat_in_ap[color] * self.zeta
                
                # Calculates the constraints.
                for color in range(2):

                    p_one_scd[color] = self.f_at_least_one(n_scd[color])

                    if color == red:
                        n_pri_zero_r[j] += f_in_bin[color] * n_halo
                        n_pri_r[j] += f_in_bin[color] * n_halo * p_one_scd[color]
                        n_scd_r[j] += f_in_bin[color] * n_halo * n_scd[color]
                        sigma_hw_r[j] += f_in_bin[color] * n_halo * p_one_scd[color] * sigma_sq[color]
                        sigma_sw_r[j] += f_in_bin[color] * n_halo * n_scd[color] * sigma_sq[color]
                    elif color == blue:
                        n_pri_zero_b[j] += f_in_bin[color] * n_halo
                        n_pri_b[j] += f_in_bin[color] * n_halo * p_one_scd[color]
                        n_scd_b[j] += f_in_bin[color] * n_halo * n_scd[color]
                        sigma_hw_b[j] += f_in_bin[color] * n_halo * p_one_scd[color] * sigma_sq[color]
                        sigma_sw_b[j] += f_in_bin[color] * n_halo * n_scd[color] * sigma_sq[color]

        nd_cen_r_out = np.array(nd_cen_r)
        nd_cen_b_out = np.array(nd_cen_b)
        nd_sat_out = np.array(nd_sat)
        n_pri_zero_r_out = self.to_output_bins(n_pri_zero_r)
        n_pri_zero_b_out = self.to_output_bins(n_pri_zero_b)
        n_pri_r_out = self.to_output_bins(n_pri_r)
        n_pri_b_out = self.to_output_bins(n_pri_b)
        n_scd_r_out = self.to_output_bins(n_scd_r)
        n_scd_b_out = self.to_output_bins(n_scd_b)
        sigma_hw_r_out = np.sqrt(self.to_output_bins(sigma_hw_r) / n_pri_r_out)
        sigma_hw_b_out = np.sqrt(self.to_output_bins(sigma_hw_b) / n_pri_b_out)
        sigma_sw_r_out = np.sqrt(self.to_output_bins(sigma_sw_r) / n_scd_r_out)
        sigma_sw_b_out = np.sqrt(self.to_output_bins(sigma_sw_b) / n_scd_b_out)
        
        return {'n_gal': nd_cen_r_out + nd_cen_b_out + nd_sat_out,
                'n_cen_r': nd_cen_r_out,
                'n_cen_b': nd_cen_b_out,
                'n_sat': nd_sat_out,
                'f_pri_r': n_pri_zero_r_out / (n_pri_zero_r_out + n_pri_zero_b_out),
                'n_mem_r': n_scd_r_out / n_pri_zero_r_out,
                'n_mem_b': n_scd_b_out / n_pri_zero_b_out,
                'sigma_sw_r': sigma_sw_r_out,
                'sigma_sw_b': sigma_sw_b_out,
                'sigma_hw_r': sigma_hw_r_out,
                'sigma_hw_b': sigma_hw_b_out,
                'log_sigma_hw_r': np.log10(sigma_hw_r_out),
                'log_sigma_hw_b': np.log10(sigma_hw_b_out),
                'r_hw_sw_sq_r': sigma_hw_r_out**2 / sigma_sw_r_out**2,
                'r_hw_sw_sq_b': sigma_hw_b_out**2 / sigma_sw_b_out**2}
    
    
    def to_output_bins(self, arr, method='Sum', weights=None):
        
        cdef int i, k
        
        if method == 'Sum':
            arr_out = np.zeros(self.n_prim_galprop_bins)
            for i in range(self.n_prim_galprop_bins_hr):
                k = int(i / self.n_prim_galprop_subbins)
                arr_out[k] += arr[i]
        
        return arr_out
