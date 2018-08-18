import sys
import getopt

import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.table import Table
import h5py
from halotools.sim_manager import CachedHaloCatalog
from scipy.stats import gaussian_kde

from satellite_kinematics import AnalyticModel, utils

DIRECTORY = None
LF_SYS_ERR = 0.02
SDSS = False

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:s', ['directory=', 'sdss'])
except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

for opt, arg in opts:
    if opt in ('-d', '--directory'):
        DIRECTORY = arg
    elif opt in ('-s', '--sdss'):
        SDSS = True

if SDSS:
    utils.a_r = 2.210
    utils.b_r = 0.478
    utils.c_r = 0.275
    utils.a_b = 2.142
    utils.b_b = 0.402
    utils.c_b = -0.170

# %% We first make a corner plot.

labels = [r'$\log L_{\rm 0, r}$', r'$\log M_{\rm 1, r}$',
          r'$\gamma_{\rm 1, r}$', r'$\gamma_{\rm 2, r}$',
          r'$\gamma_{\rm 3, r}$', r'$\sigma_{\rm r}$',
          r'$\log L_{\rm 0, b}$', r'$\log M_{\rm 1, b}$',
          r'$\gamma_{\rm 1, b}$', r'$\gamma_{\rm 2, b}$',
          r'$\gamma_{\rm 3, b}$', r'$\sigma_{\rm b}$',
          r'$f_0$', r'$\alpha_f$', r'$\beta_f$', r'$a_1$', r'$a_2$',
          r'$\log M_2$', r'$b_0$', r'$b_1$', r'$b_2$', r'$\zeta$',
          r'$\Delta\sigma_r$', r'$\Delta\sigma_b$']
labels = np.array(labels)

parameters = Table.read(DIRECTORY + 'parameters.hdf5', path='parameters')
mask = parameters['fit']
n_fp = np.sum(mask)

data = np.genfromtxt(DIRECTORY + 'ev.dat')
samples = data[:, :-3]
weights = (data[:, -3] + data[:, -2])
v_min = data[-1, -2]

data = np.genfromtxt(DIRECTORY + 'phys_live.points')
samples = np.concatenate([samples, data[:, :-2]])
weights = np.concatenate([weights, np.repeat(v_min, len(data)) + data[:, -2]])

weights = weights - np.amax(weights)
weights = np.exp(weights)

corner.corner(samples, weights=weights, plot_datapoints=False,
              plot_density=False, labels=labels[mask],
              truths=parameters['default'][mask], show_titles=True,
              levels=(0.68, 0.95, 0.997), bins=50,
              range=np.ones(np.sum(mask)) * 0.999999, title_fmt='.3f')
plt.savefig(DIRECTORY + 'posterior.pdf')
plt.savefig(DIRECTORY + 'posterior.png', dpi=300)
plt.close()


names_compact = ['log_L_0_r', 'log_M_1_r', 'gamma_2_r', 'sigma_r', 'log_L_0_b',
                 'log_M_1_b', 'gamma_2_b', 'sigma_b', 'f_0', 'alpha_f',
                 'beta_f']
mask_compact = np.zeros(np.sum(mask), dtype=np.bool)
for i in range(len(mask)):
    if mask[i]:
        if parameters['name'][i] in names_compact:
            mask_compact[np.sum(mask[:i])] = True

ndim = np.sum(mask_compact)
fig, axes = plt.subplots(ndim, ndim, figsize=(7.0, 7.0))
corner.corner(np.transpose(np.transpose(samples)[mask_compact]),
              weights=weights, plot_datapoints=False, plot_density=False,
              labels=labels[mask][mask_compact], color='royalblue',
              show_titles=False, levels=(0.68, 0.95, 0.997), bins=50,
              range=np.ones(ndim) * 0.999999, fill_contours=True, fig=fig,
              hist_kwargs={'color': 'gold', 'histtype': 'stepfilled',
                           'edgecolor': 'black', 'linewidth': 0.5},
              max_n_ticks=3, contour_kwargs={'linewidths': 0.5,
                                             'colors': 'black'})

axes = np.array(fig.axes).reshape((ndim, ndim))
for yi in range(ndim):
    for xi in range(yi + 1):
        ax = axes[yi, xi]
        ax.tick_params(axis='x', labelsize=8, rotation=90)
        ax.xaxis.set_label_coords(0.5, -0.5)
        ax.tick_params(axis='y', labelsize=8, rotation=0)
        ax.yaxis.set_label_coords(-0.5, 0.5)

for yi in range(ndim):
    ax = axes[yi, yi]
    ax.tick_params(axis='y', which='both', left='off', right='off')

plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(DIRECTORY + 'posterior_compact.pdf')
plt.savefig(DIRECTORY + 'posterior_compact.png', dpi=300)
plt.close()


tex_table = Table(names=(r'Parameter', r'Prior', r'Posterior', r'Best-fit'),
                  dtype=('S30', 'S30', 'S30', 'S30'))

for i, entry in enumerate(parameters):
    if entry['fit']:
        i_data = np.sum(parameters['fit'][:i])
        best_fit = data[:, i_data][np.argmax(data[:, -2])]
        prior_min = parameters['min'][i]
        prior_max = parameters['max'][i]
        median = corner.quantile(samples[:, i_data], 0.5, weights=weights)[0]
        lower = corner.quantile(samples[:, i_data], 0.16, weights=weights)[0]
        upper = corner.quantile(samples[:, i_data], 0.84, weights=weights)[0]
        digits = -int(np.log10(min(upper - median, median - lower))) + 2
        if (parameters['name'][i] == 'a_1' and
                not ('a_2' in parameters['name'][parameters['fit']]) and
                not ('log_M_2' in parameters['name'][parameters['fit']])):
            labels[i] = r'$\alpha_s$'
            prior_min -= 2
            prior_max -= 2
            lower -= 2
            upper -= 2
            median -= 2
        tex_table.add_row(
            [labels[i],
             '[$%.2f, %.2f$]' % (prior_min, prior_max), '$' +
             '{0:.{1}f}'.format(median, digits) + r'^{{+'
             '{0:.{1}f}'.format(upper - median, digits) + r'}' + r'_{{-'
             '{0:.{1}f}'.format(median - lower, digits) + r'}' + '$',
             '${0:.{1}f}$'.format(best_fit, digits)])

print(tex_table)

tex_table.write(DIRECTORY + 'posterior.tex', format='latex', overwrite=True)

# %% Now, we make a plot of the fit.


def chi_squared(i_min, i_max, cov, mod, dat, use):
    cov = np.copy(cov)
    use = np.copy(use)
    use = use & (np.arange(len(use)) >= i_min) & (np.arange(len(use)) < i_max)

    cov_use = []
    for i, row in enumerate(cov):
        if i < 10:
            cov[i, i] = cov[i, i] + dat[i]**2 * LF_SYS_ERR**2
        if use[i]:
            cov_use.append(row[use])
    cov_use = np.array(cov_use)
    prc = np.linalg.inv(cov_use)

    return np.inner(np.inner((mod - dat)[use], prc), (mod - dat)[use])

f = h5py.File(DIRECTORY + 'parameters.hdf5', "r")
lf_sys_err = f.attrs['lf_sys_err']
covariance = f.attrs['covariance']
bias = f.attrs['bias']
halocat = f.attrs['halocat']
profile = f.attrs['profile']
catalog = f.attrs['catalog']
f.close()

if halocat == 'smdpl':
    halocat = CachedHaloCatalog(simname='smdpl', redshift=utils.z_halo,
                                dz_tol=0.01)
elif halocat == 'conplanck':
    halocat = CachedHaloCatalog(simname='conplanck', redshift=utils.z_halo,
                                version_name='1', dz_tol=0.01)

analytic_model = AnalyticModel(halocat, threshold=9.5, profile=profile,
                               a_r=utils.a_r, b_r=utils.b_r, c_r=utils.c_r,
                               a_b=utils.a_b, b_b=utils.b_b, c_b=utils.c_b)
default_param_dict = analytic_model.param_dict
analytic_model.param_dict = utils.get_best_fit_param_dict(DIRECTORY)
model = analytic_model.get_constraints_dict()

catalog = utils.read_catalog_from_file(catalog)
sdss = utils.constraints_dict_from_spec_catalog(catalog, profile=profile)
use = ~np.isnan(utils.constraints_dict_to_vector(sdss))

f = h5py.File(covariance + 'constraints.hdf5', 'r')
n_s = f.attrs['n_mocks']
use = use & f['data_use'][:]
sdss_cov = np.array(f['data_cov'])
sdss_err = utils.constraints_vector_to_dict(np.sqrt(np.diag(sdss_cov)))
f.close()

f = h5py.File(bias + 'constraints.hdf5', 'r')
use = use & f['data_use'][:]
bias = f['data_bias'][:]
bias[:10] = 0.0

bias = utils.constraints_vector_to_dict(bias)
use = utils.constraints_vector_to_dict(use)

for key in use.keys():
    sdss[key][~use[key]] = np.nan
    model[key][~use[key]] = np.nan
    sdss_err[key][~use[key]] = np.nan

plt.figure(figsize=(7.0, 4.5))
ax1 = plt.subplot2grid(shape=(2, 6), loc=(1, 0), colspan=2)
ax2 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax3 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
ax4 = plt.subplot2grid((2, 6), (0, 1), colspan=2)
ax5 = plt.subplot2grid((2, 6), (0, 3), colspan=2)

log_lum = 0.5 * (utils.default_log_lum_bins[1:] +
                 utils.default_log_lum_bins[:-1])

chi = np.array([chi_squared(
        k * 10, (k + 1) * 10, sdss_cov,
        utils.constraints_dict_to_vector(model) +
        utils.constraints_dict_to_vector(bias),
        utils.constraints_dict_to_vector(sdss),
        utils.constraints_dict_to_vector(use)) for k in range(8)])

ax1.plot(log_lum, bias['n_mem_r'] + model['n_mem_r'], color='tomato')
ax1.errorbar(log_lum, sdss['n_mem_r'], yerr=sdss_err['n_mem_r'], fmt='x',
             color='tomato')
ax1.annotate(r'$\chi^2 = %.1f$' % chi[2], xy=(0.05, 0.95),
             xycoords='axes fraction', verticalalignment='top', color='tomato')
ax1.plot(log_lum + 0.02, bias['n_mem_b'] + model['n_mem_b'], color='royalblue')
ax1.errorbar(log_lum + 0.02, sdss['n_mem_b'], yerr=sdss_err['n_mem_b'],
             fmt='x', color='royalblue')
ax1.annotate(r'$\chi^2 = %.1f$' % chi[3], xy=(0.05, 0.85),
             xycoords='axes fraction', verticalalignment='top',
             color='royalblue')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')
ax1.set_ylabel(r'$n_{\rm sat}$')
ax1.set_xlim(utils.default_log_lum_bins[0] - 0.1,
             utils.default_log_lum_bins[-1] + 0.1)

ax2.plot(log_lum, bias['log_sigma_hw_r'] + model['log_sigma_hw_r'],
         color='tomato')
ax2.errorbar(log_lum, sdss['log_sigma_hw_r'], yerr=sdss_err['log_sigma_hw_r'],
             fmt='x', color='tomato')
ax2.annotate(r'$\chi^2 = %.1f$' % chi[4], xy=(0.05, 0.95),
             xycoords='axes fraction', verticalalignment='top', color='tomato')
ax2.plot(log_lum + 0.02, bias['log_sigma_hw_b'] + model['log_sigma_hw_b'],
         color='royalblue')
ax2.errorbar(log_lum + 0.02, sdss['log_sigma_hw_b'],
             yerr=sdss_err['log_sigma_hw_b'], fmt='x', color='royalblue')
ax2.annotate(r'$\chi^2 = %.1f$' % chi[6], xy=(0.05, 0.85),
             xycoords='axes fraction', verticalalignment='top',
             color='royalblue')
ax2.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')
ax2.set_ylabel(r'$\log \sigma_{\rm hw}$')
ax2.set_xlim(utils.default_log_lum_bins[0] - 0.1,
             utils.default_log_lum_bins[-1] + 0.1)

ax3.plot(log_lum, bias['r_hw_sw_sq_r'] + model['r_hw_sw_sq_r'],
         color='tomato')
ax3.errorbar(log_lum, sdss['r_hw_sw_sq_r'], yerr=sdss_err['r_hw_sw_sq_r'],
             fmt='x', color='tomato')
ax3.annotate(r'$\chi^2 = %.1f$' % chi[5], xy=(0.05, 0.15),
             xycoords='axes fraction', color='tomato')
ax3.plot(log_lum + 0.02, bias['r_hw_sw_sq_b'] + model['r_hw_sw_sq_b'],
         color='royalblue')
ax3.errorbar(log_lum + 0.02, sdss['r_hw_sw_sq_b'],
             yerr=sdss_err['r_hw_sw_sq_b'], fmt='x', color='royalblue')
ax3.annotate(r'$\chi^2 = %.1f$' % chi[7], xy=(0.05, 0.05),
             xycoords='axes fraction', color='royalblue')
ax3.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')
ax3.set_ylabel(r'$\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2$')
ax3.set_xlim(utils.default_log_lum_bins[0] - 0.1,
             utils.default_log_lum_bins[-1] + 0.1)

ax4.plot(log_lum, bias['n_gal'] + model['n_gal'], color='black')
ax4.errorbar(log_lum, sdss['n_gal'], yerr=sdss_err['n_gal'], fmt='x',
             color='black')
ax4.annotate(r'$\chi^2 = %.1f$' % chi[0], xy=(0.05, 0.05),
             xycoords='axes fraction', verticalalignment='bottom')
ax4.set_yscale('log')
ax4.set_xlabel(r'$\log \ L \ [h^{-2} L_\odot]$')
ax4.set_ylabel(r'$n_{\rm gal}$')
ax4.set_xlim(utils.default_log_lum_bins[0] - 0.1,
             utils.default_log_lum_bins[-1] + 0.1)

ax5.plot(log_lum, bias['f_pri_r'] + model['f_pri_r'], color='black')
ax5.errorbar(log_lum, sdss['f_pri_r'], yerr=sdss_err['f_pri_r'], fmt='x',
             color='black')
ax5.annotate(r'$\chi^2 = %.1f$' % chi[1], xy=(0.05, 0.95),
             xycoords='axes fraction', verticalalignment='top')
ax5.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')
ax5.set_ylabel(r'$f_{\rm pri, r}$')
ax5.set_xlim(utils.default_log_lum_bins[0] - 0.1,
             utils.default_log_lum_bins[-1] + 0.1)

plt.tight_layout(pad=0.3)
plt.savefig(DIRECTORY + 'constraints.pdf')
plt.savefig(DIRECTORY + 'constraints.png', dpi=300)
plt.close()

n_d = np.sum(utils.constraints_dict_to_vector(use))
print("Best-fit chi squared: %.3f" % (chi_squared(
    0, 80, sdss_cov, utils.constraints_dict_to_vector(model) +
    utils.constraints_dict_to_vector(bias),
    utils.constraints_dict_to_vector(sdss),
    utils.constraints_dict_to_vector(use)) *
    float(n_s - n_d - 2) / float(n_s - 1)))
print("Degrees of freedom: %d - %d" % (n_d, n_fp))

# %% Let's now look at posterior distributions.

param_dict_sample = utils.get_param_dict_sample(DIRECTORY)

# We start with the satellite fraction.

analytic_model.param_dict = default_param_dict
constraints = analytic_model.get_constraints_dict()

f_sat_default = np.sum(constraints['n_sat']) / np.sum(constraints['n_gal'])

f_sat_sample = np.zeros(len(param_dict_sample))
for i in range(len(param_dict_sample)):
    analytic_model.param_dict = param_dict_sample[i]
    constraints = analytic_model.get_constraints_dict()
    f_sat_sample[i] = (np.sum(constraints['n_sat']) /
                       np.sum(constraints['n_gal']))

plt.hist(f_sat_sample, normed=True, histtype='step', bins=100)
plt.title(r'$f_{\rm sat} = %.3f_{-%.3f}^{+%.3f}$' % (
    np.percentile(f_sat_sample, 50),
    np.percentile(f_sat_sample, 50) - np.percentile(f_sat_sample, 16),
    np.percentile(f_sat_sample, 84) - np.percentile(f_sat_sample, 50)))
plt.axvline(f_sat_default, ls='--')
plt.xlabel(r'Satellite fraction $f_{\rm sat}$')
plt.ylabel(r'Posterior')
plt.tight_layout(pad=0.3)
plt.savefig(DIRECTORY + 'f_sat.pdf')
plt.savefig(DIRECTORY + 'f_sat.png', dpi=300)
plt.close()

# Next, we make a plot of the entire galaxy-halo connection.

log_mvir_bins = np.linspace(10.5, 14.5, 100)

fig, axarr = plt.subplots(2, 2, figsize=(3.33, 3.33))

model = utils.default_model(threshold=9.5)

model.param_dict = default_param_dict
f_r_default = model.mean_red_fraction_centrals(prim_haloprop=10**log_mvir_bins)
log_L_r_default = np.log10(model.median_prim_galprop_red_centrals(
    prim_haloprop=10**log_mvir_bins))
log_L_b_default = np.log10(model.median_prim_galprop_blue_centrals(
    prim_haloprop=10**log_mvir_bins))
log_n_sat_default = np.log10(model.mean_occupation_satellites(
    prim_haloprop=10**log_mvir_bins))

f_r_sample = np.zeros((len(param_dict_sample), len(log_mvir_bins)))
log_L_r_sample = np.zeros((len(param_dict_sample), len(log_mvir_bins)))
log_L_b_sample = np.zeros((len(param_dict_sample), len(log_mvir_bins)))
log_n_sat_sample = np.zeros((len(param_dict_sample), len(log_mvir_bins)))
for i in range(len(param_dict_sample)):
    model.param_dict = param_dict_sample[i]
    f_r_sample[i, :] = model.mean_red_fraction_centrals(
        prim_haloprop=10**log_mvir_bins)
    log_L_r_sample[i, :] = np.log10(model.median_prim_galprop_red_centrals(
        prim_haloprop=10**log_mvir_bins))
    log_L_b_sample[i, :] = np.log10(model.median_prim_galprop_blue_centrals(
        prim_haloprop=10**log_mvir_bins))
    log_n_sat_sample[i, :] = np.log10(model.mean_occupation_satellites(
        prim_haloprop=10**log_mvir_bins))

axarr[0, 0].fill_between(
    log_mvir_bins, np.percentile(log_L_r_sample, 16, axis=0),
    np.percentile(log_L_r_sample, 84, axis=0), color='tomato', alpha=0.5,
    label=r'68%%', lw=0)
axarr[0, 0].fill_between(
    log_mvir_bins, np.percentile(log_L_r_sample, 2.5, axis=0),
    np.percentile(log_L_r_sample, 97.5, axis=0), color='tomato', alpha=0.25,
    label=r'95%%', lw=0)
axarr[0, 0].fill_between(
    log_mvir_bins, np.percentile(log_L_b_sample, 16, axis=0),
    np.percentile(log_L_b_sample, 84, axis=0), color='royalblue', alpha=0.5,
    label=r'68%%', lw=0)
axarr[0, 0].fill_between(
    log_mvir_bins, np.percentile(log_L_b_sample, 2.5, axis=0),
    np.percentile(log_L_b_sample, 97.5, axis=0), color='royalblue', alpha=0.25,
    label=r'95%%', lw=0)
axarr[0, 0].set_xticks([11, 12, 13, 14])
axarr[0, 0].set_yticks([9.5, 10, 10.5, 11, 11.5])
axarr[0, 0].set_xlim(np.amin(log_mvir_bins), np.amax(log_mvir_bins))
axarr[0, 0].set_ylim(9.4, max(np.percentile(log_L_r_sample, 99, axis=0)[-1],
                              np.percentile(log_L_b_sample, 99, axis=0)[-1]))
axarr[0, 0].set_xlabel(r'$\log M_{\rm vir} \ [h^{-1} M_\odot]$')
axarr[0, 0].set_ylabel(r'$\langle \log L_c \rangle \ [h^{-2} L_\odot]$')
axarr[0, 0].xaxis.set_ticks_position('top')
axarr[0, 0].xaxis.set_label_position('top')
axarr[0, 0].yaxis.set_ticks_position('left')
axarr[0, 0].yaxis.set_label_position('left')
for tick in axarr[0, 0].get_yticklabels():
    tick.set_rotation(90)
axarr[0, 0].tick_params(direction='in', pad=2)

axarr[1, 0].fill_between(log_mvir_bins, np.percentile(f_r_sample, 16, axis=0),
                         np.percentile(f_r_sample, 84, axis=0), color='grey',
                         alpha=0.5, label=r'68%%', lw=0)
axarr[1, 0].fill_between(log_mvir_bins, np.percentile(f_r_sample, 2.5, axis=0),
                         np.percentile(f_r_sample, 97.5, axis=0),
                         color='grey', alpha=0.25, label=r'95%%', lw=0)
axarr[1, 0].set_xticks([11, 12, 13, 14])
axarr[1, 0].set_xlim(np.amin(log_mvir_bins), np.amax(log_mvir_bins))
axarr[1, 0].set_ylim(0, 1.1)
axarr[1, 0].set_xlabel(r'$\log M_{\rm vir} \ [h^{-1} M_\odot]$')
axarr[1, 0].set_ylabel(r'$f_r$')
axarr[1, 0].xaxis.set_ticks_position('bottom')
axarr[1, 0].xaxis.set_label_position('bottom')
axarr[1, 0].yaxis.set_ticks_position('left')
axarr[1, 0].yaxis.set_label_position('left')
for tick in axarr[1, 0].get_yticklabels():
    tick.set_rotation(90)
axarr[1, 0].tick_params(direction='in', pad=2)

sigma_r = np.array([param_dict_sample[i]['sigma_r'] for i in
                    range(len(param_dict_sample))])
sigma_b = np.array([param_dict_sample[i]['sigma_b'] for i in
                    range(len(param_dict_sample))])

bins = np.linspace(min(np.percentile(sigma_r, 0.01),
                       np.percentile(sigma_b, 0.01)),
                   max(np.percentile(sigma_r, 99.99),
                       np.percentile(sigma_b, 99.99)), 500)

density = gaussian_kde(sigma_r)
axarr[0, 1].plot(bins, density(bins), color='tomato')
density = gaussian_kde(sigma_b)
axarr[0, 1].plot(bins, density(bins), color='royalblue')
axarr[0, 1].set_xlim(np.amin(bins), np.amax(bins))
axarr[0, 1].set_ylim(bottom=0)
axarr[0, 1].set_xlabel(r'$\sigma \ [\mathrm{dex}]$')
axarr[0, 1].set_ylabel(r'$dp / d\sigma \ [\mathrm{dex}^{-1}]$')
axarr[0, 1].xaxis.set_ticks_position('top')
axarr[0, 1].xaxis.set_label_position('top')
axarr[0, 1].yaxis.set_ticks_position('right')
axarr[0, 1].yaxis.set_label_position('right')
for tick in axarr[0, 1].get_yticklabels():
    tick.set_rotation(90)
axarr[0, 1].tick_params(direction='in', pad=2)

axarr[1, 1].fill_between(
    log_mvir_bins, np.percentile(log_n_sat_sample, 16, axis=0),
    np.percentile(log_n_sat_sample, 84, axis=0), color='grey', alpha=0.5,
    label=r'68%%', lw=0)
axarr[1, 1].fill_between(
    log_mvir_bins, np.percentile(log_n_sat_sample, 2.5, axis=0),
    np.percentile(log_n_sat_sample, 97.5, axis=0), color='grey',
    alpha=0.25, label=r'95%%', lw=0)
axarr[1, 1].set_xticks([11, 12, 13, 14])
axarr[1, 1].set_xlim(np.amin(log_mvir_bins), np.amax(log_mvir_bins))
axarr[1, 1].set_ylim(-2, np.percentile(log_n_sat_sample, 99, axis=0)[-1])
axarr[1, 1].set_xlabel(r'$\log M_{\rm vir} \ [h^{-1} M_\odot]$')
axarr[1, 1].set_ylabel(r'$\log \langle N_{\rm sat} \rangle$')
axarr[1, 1].xaxis.set_ticks_position('bottom')
axarr[1, 1].xaxis.set_label_position('bottom')
axarr[1, 1].yaxis.set_ticks_position('right')
axarr[1, 1].yaxis.set_label_position('right')
for tick in axarr[1, 1].get_yticklabels():
    tick.set_rotation(90)
axarr[1, 1].tick_params(direction='in', pad=2)

plt.tight_layout(pad=0.25)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig(DIRECTORY + 'model_post.pdf')
plt.savefig(DIRECTORY + 'model_post.png', dpi=300)

axarr[0, 0].plot(log_mvir_bins, log_L_r_default, color='tomato', ls='--')
axarr[0, 0].plot(log_mvir_bins, log_L_b_default, color='royalblue', ls='--')
axarr[0, 1].axvline(default_param_dict['sigma_r'], color='tomato', ls='--')
axarr[0, 1].axvline(default_param_dict['sigma_b'], color='royalblue', ls='--')
axarr[1, 0].plot(log_mvir_bins, f_r_default, color='black', ls='--')
axarr[1, 1].plot(log_mvir_bins, log_n_sat_default, color='black', ls='--')

plt.savefig(DIRECTORY + 'model_post_vs_default.pdf')
plt.savefig(DIRECTORY + 'model_post_vs_default.png', dpi=300)
plt.close()

mask = np.array(['d_sigma' in s for s in parameters['name']])
if np.all(parameters['fit'][mask]):
    sigma_r = np.array([param_dict_sample[i]['sigma_r'] for i in
                        range(len(param_dict_sample))])
    sigma_b = np.array([param_dict_sample[i]['sigma_b'] for i in
                        range(len(param_dict_sample))])
    d_sigma_r = np.array([param_dict_sample[i]['d_sigma_r'] for i in
                          range(len(param_dict_sample))])
    d_sigma_b = np.array([param_dict_sample[i]['d_sigma_b'] for i in
                          range(len(param_dict_sample))])

    log_mvir = np.linspace(12, 14.5, 100)

    s_r_min = np.zeros_like(log_mvir)
    s_r_max = np.zeros_like(log_mvir)
    s_b_min = np.zeros_like(log_mvir)
    s_b_max = np.zeros_like(log_mvir)

    for i in range(len(log_mvir)):
        s_r = sigma_r + d_sigma_r * (log_mvir[i] - 14)
        s_b = sigma_b + d_sigma_b * (log_mvir[i] - 12)

        s_r_min[i] = np.percentile(s_r, 16)
        s_r_max[i] = np.percentile(s_r, 84)
        s_b_min[i] = np.percentile(s_b, 16)
        s_b_max[i] = np.percentile(s_b, 84)

    plt.fill_between(log_mvir, s_r_min, s_r_max, color='tomato', alpha=0.5)
    plt.fill_between(log_mvir, s_b_min, s_b_max, color='royalblue', alpha=0.5)
    plt.xlabel(r'Halo Mass $\log M_{\rm vir} \ [h^{-1} M_\odot]$')
    plt.ylabel(r'Scatter $\sigma (M_{\rm vir}) [\mathrm{dex}]$')
    plt.xlim(np.amin(log_mvir), np.amax(log_mvir))
    plt.ylim(bottom=0)
    plt.tight_layout(pad=0.3)
    plt.savefig(DIRECTORY + 'scatter_vs_mass.pdf')
    plt.savefig(DIRECTORY + 'scatter_vs_mass.png', dpi=300)
    plt.close()
