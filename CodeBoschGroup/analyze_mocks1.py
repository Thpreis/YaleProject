import sys
import getopt
from mpi4py import MPI
import time

import h5py
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from halotools.sim_manager import CachedHaloCatalog
from scipy.stats.mstats import normaltest

from satellite_kinematics import utils, AnalyticModel

DIRECTORY = None
I_JOB = MPI.COMM_WORLD.Get_rank()
N_JOB = MPI.COMM_WORLD.Get_size()
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

f = h5py.File(DIRECTORY + 'parameters.hdf5', "r")
param_dict = dict(f.attrs)
profile = param_dict.pop('profile', None)
if profile == 'subhalo':
    profile = 3
halocat = param_dict.pop('halocat', None)
n_mocks = param_dict.pop('n_mocks', None)
param_dict['zeta'] = 1.0
param_dict['d_sigma_r'] = 0.0
param_dict['d_sigma_b'] = 0.0
f.close()

# %% First, we run our analysis pipeline on all mocks.

use = np.zeros(n_mocks, dtype=np.bool)
use[I_JOB::N_JOB] = True

if I_JOB == 0:

    print("Analyzing mock catalogs...")
    print("Halo catalog: %s" % halocat)
    print("Radial profile: %d" % profile)
    print("Number of mocks: %d" % n_mocks)

    for key in param_dict.keys():
        print("%s: %.3f" % (key, param_dict[key]))

else:
    time.sleep(1)

for i in np.arange(n_mocks)[use]:
    try:
        Table.read(DIRECTORY + 'constraint_%i.hdf5' % i, path='constraints')
    except IOError:
        print("Working on sample %d" % i)
        catalog = utils.read_catalog_from_file(DIRECTORY + 'mock_%d.hdf5' % i)
        constraints = utils.constraints_dict_from_spec_catalog(
            catalog, profile=profile)
        utils.write_constraints_to_file(
            constraints, DIRECTORY + 'constraint_%i.hdf5' % i, 'constraints')

# %% Then, we make some summary plots.

if I_JOB != 0:
    sys.exit()

vector_all = []

while True:
    try:
        for i in range(n_mocks):
            vector_all.append(utils.constraints_dict_to_vector(
                Table.read(DIRECTORY + 'constraint_%d.hdf5' % i,
                           path='constraints')))
        break
    except IOError:
        time.sleep(10)

vector_all = np.array(vector_all)

cov = np.cov(vector_all.T)
cor = np.corrcoef(vector_all.T)
ave = np.mean(vector_all, axis=0)
use = np.ones(len(ave), dtype=np.bool)
for i in range(len(ave)):
    use[i] = np.all(~np.isnan(vector_all[:, i]))
for i in range(50, 60):
    use[i] = np.mean(vector_all[:, i - 30]) > 1e-1
for i in range(70, 80):
    use[i] = np.mean(vector_all[:, i - 40]) > 1e-1

names = [r'$n_{\rm gal}$', r'$f_{\rm pri, r}$',
         r'$n_{\rm s, r}$', r'$n_{\rm s, b}$',
         r'$\log \sigma_{\rm hw, r}$',
         r'$(\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2)_{\rm r}$',
         r'$\log \sigma_{\rm hw, b}$',
         r'$(\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2)_{\rm b}$']

fig, axarr = plt.subplots(8, 10, figsize=(11.0, 8.5))

for i in range(vector_all.shape[1]):
    if use[i]:
        axarr[i // 10, i % 10].hist(vector_all[:, i], bins=30)
        axarr[i // 10, i % 10].annotate(
            '%.2e' % normaltest(vector_all[:, i])[1], xy=(0.05, 0.95),
            xycoords='axes fraction', verticalalignment='top')
    if i % 10 == 0:
        axarr[i // 10, i % 10].set_ylabel(names[i // 10])
    axarr[i // 10, i % 10].tick_params(
        bottom='off', top='off', left='off', right='off', labelbottom='off',
        labelleft='off')

fig.subplots_adjust(hspace=0.0, wspace=0.0)
plt.tight_layout(pad=0.3)
plt.savefig(DIRECTORY + 'variance.pdf')
plt.savefig(DIRECTORY + 'variance.png', dpi=300)
plt.close()


cov = np.cov(vector_all.T)
cor = np.corrcoef(vector_all.T)

for i in range(vector_all.shape[1]):
    if not use[i]:
        cor[i, :] = np.nan
        cor[:, i] = np.nan

for i in range(1, 9):
    plt.axvline(i * 10 - 0.5, ls='--', color='black', lw=0.5)
    plt.axhline(i * 10 - 0.5, ls='--', color='black', lw=0.5)

plt.imshow(cor, vmin=-1.0, vmax=1.0, cmap=plt.get_cmap('RdBu'))
plt.xticks(np.linspace(5, 75, 8) - 0.5, names, rotation='vertical')
plt.yticks(np.linspace(5, 75, 8) - 0.5, names)
cb = plt.colorbar()
cb.set_label('Correlation')
plt.tight_layout(pad=0.3)
plt.savefig(DIRECTORY + 'covariance.pdf')
plt.savefig(DIRECTORY + 'covariance.png', dpi=300)
plt.close()


plt.figure(figsize=(3.33, 4.0))
cor_compact = np.zeros((np.sum(use), np.sum(use)))
for i in range(cor.shape[0]):
    if not np.isnan(cor[i, i]):
        cor_compact[np.sum(use[:i])] = cor[i, :][~np.isnan(cor[i, :])]

for i in range(1, 9):
    plt.axvline(np.sum(use[:10*i]) - 0.5, ls='--', color='black', lw=0.5)
    plt.axhline(np.sum(use[:10*i]) - 0.5, ls='--', color='black', lw=0.5)

names = [r'$n_{\rm gal}$', r'$f_{\rm pri, r}$',
         r'$n_{\rm s}$', r'$n_{\rm s}$',
         r'$\log \sigma_{\rm hw}$',
         r'$\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2$',
         r'$\log \sigma_{\rm hw}$',
         r'$\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2$']
ticks = np.zeros(8)
ax = plt.gca()
ax.xaxis.tick_top()
colors = ['black', 'black', 'tomato', 'royalblue', 'tomato', 'tomato',
          'royalblue', 'royalblue']
for i in range(8):
    ticks[i] = 0.5 * np.sum(use[:10*i]) + 0.5 * np.sum(use[:10*(i+1)])
plt.xticks(ticks, names, rotation=90)
plt.yticks(ticks, names)
for xtick, color in zip(ax.get_xticklabels(), colors):
    xtick.set_color(color)
for ytick, color in zip(ax.get_yticklabels(), colors):
    ytick.set_color(color)

plt.imshow(cor_compact, vmin=-1.0, vmax=1.0, cmap=plt.get_cmap('PuOr'))
cb = plt.colorbar(orientation='horizontal', fraction=0.047, pad=0.02,
                  ticks=[-1, 0, 1])
cb.set_label('Correlation')
plt.tick_params(axis=u'both', which=u'both', length=0)
plt.tight_layout(pad=0.2)
plt.savefig(DIRECTORY + 'covariance_compact.pdf')
plt.savefig(DIRECTORY + 'covariance_compact.png', dpi=300)


f = h5py.File(DIRECTORY + 'constraints.hdf5', 'w')
f.attrs['n_mocks'] = n_mocks

try:
    f.create_dataset("data_use", data=use)
    f.create_dataset("data_ave", data=ave)
    f.create_dataset("data_cov", data=cov)
except RuntimeError:
    f['data_use'][:] = use
    f['data_ave'][:] = ave
    f['data_cov'][:] = cov

if halocat == 'smdpl':
    halocat = CachedHaloCatalog(simname='smdpl', redshift=utils.z_halo,
                                dz_tol=0.01)
elif halocat == 'conplanck':
    halocat = CachedHaloCatalog(simname='conplanck', redshift=utils.z_halo,
                                dz_tol=0.01, version_name='1')

model = AnalyticModel(halocat, threshold=9.5, profile=profile,
                      a_r=utils.a_r, b_r=utils.b_r, c_r=utils.c_r,
                      a_b=utils.a_b, b_b=utils.b_b, c_b=utils.c_b)
model.param_dict = param_dict
constraints_analytic = utils.constraints_dict_to_vector(
    model.get_constraints_dict())

bias = ave - constraints_analytic

try:
    f.create_dataset("data_bias", data=bias)
except RuntimeError:
    f['data_bias'][:] = bias

f.close()

constraints_analytic = np.where(np.isnan(ave), np.nan, constraints_analytic)
constraints_analytic = np.where(~use, np.nan, constraints_analytic)
ave = np.where(~use, np.nan, ave)
bias = np.where(~use, np.nan, bias)

plt.figure(figsize=(7.0, 4.5))
ax1 = plt.subplot2grid(shape=(2, 6), loc=(1, 0), colspan=2)
ax2 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax3 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
ax4 = plt.subplot2grid((2, 6), (0, 1), colspan=2)
ax5 = plt.subplot2grid((2, 6), (0, 3), colspan=2)


def add_subplot_axes(ax, rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

log_lum = 0.5 * (utils.default_log_lum_bins[1:] +
                 utils.default_log_lum_bins[:-1])

ax1.errorbar(log_lum, constraints_analytic[20:30] / ave[20:30],
             yerr=(np.sqrt(np.diag(cov)[20:30]) / ave[20:30]),
             color='tomato', fmt='x')
ax1.errorbar(log_lum + 0.02, constraints_analytic[30:40] / ave[30:40],
             yerr=(np.sqrt(np.diag(cov)[30:40]) / ave[30:40]),
             color='royalblue', fmt='x')
ax1.axhline(1.0, ls='--', color='black')
ax1.set_ylabel(r'$n_{\rm s, m} / n_{\rm s, f}$')
ax1.set_xlim(np.amin(utils.default_log_lum_bins),
             np.amax(utils.default_log_lum_bins))
ax1.set_ylim(0.3, 2.4)
ax1.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')

ax2.errorbar(log_lum, -bias[40:50], yerr=np.sqrt(np.diag(cov)[40:50]),
             color='tomato', fmt='x')
ax2.errorbar(log_lum + 0.02, -bias[60:70], yerr=np.sqrt(np.diag(cov)[60:70]),
             color='royalblue', fmt='x')
ax2.axhline(0.0, ls='--', color='black')
ax2.set_ylabel(r'$\log \sigma_{\rm hw, m} / \sigma_{\rm hw, f}$')
ax2.set_xlim(np.amin(utils.default_log_lum_bins),
             np.amax(utils.default_log_lum_bins))
ax2.set_ylim(-0.25, 0.5)
ax2.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')

ax3.errorbar(log_lum, -bias[50:60], yerr=np.sqrt(np.diag(cov)[50:60]),
             color='tomato', fmt='x')
ax3.errorbar(log_lum + 0.02, -bias[70:80], yerr=np.sqrt(np.diag(cov)[70:80]),
             color='royalblue', fmt='x')
ax3.axhline(0.0, ls='--', color='black')
ax3.set_ylabel(r'$(\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2)_{\rm m}$' +
               r'$- (\sigma_{\rm hw}^2 / \sigma_{\rm sw}^2)_{\rm f}$')
ax3.set_xlim(np.amin(utils.default_log_lum_bins),
             np.amax(utils.default_log_lum_bins))
ax3.set_ylim(-0.25, 0.5)
ax3.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')

ax4.errorbar(log_lum, constraints_analytic[0:10] / ave[0:10],
             yerr=(np.sqrt(np.diag(cov)[0:10]) / ave[0:10]),
             color='black', fmt='x')
ax4.axhline(1.0, ls='--', color='black')
ax4.set_ylabel(r'$n_{\rm gal, m} / n_{\rm gal, f}$')
ax4.set_ylim(0.8, 1.4)
ax4.set_xlabel(r'$\log \ L \ [h^{-2} L_\odot]$')

ax5.errorbar(log_lum, -bias[10:20], yerr=np.sqrt(np.diag(cov)[10:20]),
             color='black', fmt='x')
ax5.axhline(0.0, ls='--', color='black')
ax5.set_ylabel(r'$f_{\rm pri, r, m} - f_{\rm pri, r, f}$')
ax5.set_ylim(-0.05, 0.10)
ax5.set_xlabel(r'$\log \ L_{\rm pri} \ [h^{-2} L_\odot]$')

plt.tight_layout(pad=0.4)

rect = [0.05, 0.55, 0.4, 0.4]
ax1_sub = add_subplot_axes(ax1, rect)
ax1_sub.plot(log_lum, ave[20:30], color='tomato', ls='--')
ax1_sub.plot(log_lum, constraints_analytic[20:30], color='tomato')
ax1_sub.plot(log_lum, ave[30:40], color='royalblue', ls='--')
ax1_sub.plot(log_lum, constraints_analytic[30:40], color='royalblue')
ax1_sub.xaxis.set_visible(False)
ax1_sub.yaxis.set_ticks_position('right')
ax1_sub.set_xlim(np.amin(utils.default_log_lum_bins),
                 np.amax(utils.default_log_lum_bins))
ax1_sub.set_yscale('log')

rect = [0.05, 0.55, 0.4, 0.4]
ax2_sub = add_subplot_axes(ax2, rect)
ax2_sub.plot(log_lum, ave[40:50], color='tomato', ls='--')
ax2_sub.plot(log_lum, constraints_analytic[40:50], color='tomato')
ax2_sub.plot(log_lum, ave[60:70], color='royalblue', ls='--')
ax2_sub.plot(log_lum, constraints_analytic[60:70], color='royalblue')
ax2_sub.xaxis.set_visible(False)
ax2_sub.yaxis.set_ticks_position('right')
ax2_sub.set_xlim(np.amin(utils.default_log_lum_bins),
                 np.amax(utils.default_log_lum_bins))

rect = [0.05, 0.55, 0.4, 0.4]
ax3_sub = add_subplot_axes(ax3, rect)
ax3_sub.plot(log_lum, ave[50:60], color='tomato', ls='--')
ax3_sub.plot(log_lum, constraints_analytic[50:60], color='tomato')
ax3_sub.plot(log_lum, ave[70:80], color='royalblue', ls='--')
ax3_sub.plot(log_lum, constraints_analytic[70:80], color='royalblue')
ax3_sub.xaxis.set_visible(False)
ax3_sub.yaxis.set_ticks_position('right')
ax3_sub.set_xlim(np.amin(utils.default_log_lum_bins),
                 np.amax(utils.default_log_lum_bins))

rect = [0.05, 0.55, 0.4, 0.4]
ax4_sub = add_subplot_axes(ax4, rect)
ax4_sub.plot(log_lum, ave[0:10], color='black', ls='--')
ax4_sub.plot(log_lum, constraints_analytic[0:10], color='black')
ax4_sub.xaxis.set_visible(False)
ax4_sub.yaxis.set_ticks_position('right')
ax4_sub.set_yscale('log')

rect = [0.05, 0.55, 0.4, 0.4]
ax5_sub = add_subplot_axes(ax5, rect)
ax5_sub.plot(log_lum, ave[10:20], color='black', ls='--')
ax5_sub.plot(log_lum, constraints_analytic[10:20], color='black')
ax5_sub.xaxis.set_visible(False)
ax5_sub.yaxis.set_ticks_position('right')

plt.savefig(DIRECTORY + 'bias.pdf')
plt.savefig(DIRECTORY + 'bias.png', dpi=300)

fig, axarr = plt.subplots(2, 4, figsize=(7.0, 4.5), sharex=True)
log_lum = 0.5 * (utils.default_log_lum_bins[1:] +
                 utils.default_log_lum_bins[:-1])

axarr[0, 0].plot(log_lum, np.sqrt(np.diag(cov)[0:10]) / ave[0:10],
                 color='black')
axarr[0, 0].set_ylabel(r'$\Delta_{n_{\rm gal}} / n_{\rm gal}$')
axarr[0, 0].set_ylim(0, 0.2)

axarr[1, 0].plot(log_lum, np.sqrt(np.diag(cov)[10:20]), color='black')
axarr[1, 0].set_ylabel(r'$\Delta_{f_{\rm pri, r}}$')
axarr[1, 0].set_ylim(0, 0.1)

axarr[0, 1].plot(log_lum, np.sqrt(np.diag(cov)[20:30]) / ave[20:30],
                 color='black')
axarr[0, 1].set_ylabel(r'$\Delta_{n_{\rm mem, r}} / n_{\rm mem, r}$')
axarr[0, 1].set_ylim(0, 0.3)

axarr[1, 1].plot(log_lum, np.sqrt(np.diag(cov)[30:40]) / ave[30:40],
                 color='black')
axarr[1, 1].set_ylabel(r'$\Delta_{n_{\rm mem, b}} / n_{\rm mem, b}$')
axarr[1, 1].set_ylim(0, 0.3)

axarr[0, 2].plot(log_lum,  np.sqrt(np.diag(cov)[40:50]), color='black')
axarr[0, 2].set_ylabel(r'$\Delta_{\log \sigma_{\rm hw, r}}$')
axarr[0, 2].set_ylim(0.0, 0.3)

axarr[1, 2].plot(log_lum,  np.sqrt(np.diag(cov)[60:70]), color='black')
axarr[1, 2].set_ylabel(r'$\Delta_{\log \sigma_{\rm hw, b}}$')
axarr[1, 2].set_ylim(0.0, 0.3)

axarr[0, 3].plot(log_lum,  np.sqrt(np.diag(cov)[50:60]), color='black')
axarr[0, 3].set_ylabel(r'$\Delta_{r^2_{\rm r}}$')
axarr[0, 3].set_ylim(0.0, 0.3)

axarr[1, 3].plot(log_lum,  np.sqrt(np.diag(cov)[70:80]), color='black')
axarr[1, 3].set_ylabel(r'$\Delta_{r^2_{\rm b}}$')
axarr[1, 3].set_ylim(0.0, 0.3)

axarr[1, 0].set_xlabel(r'$\log \ L_{\rm pri} \ [L_\odot h^{-2}]$')
axarr[1, 1].set_xlabel(r'$\log \ L_{\rm pri} \ [L_\odot h^{-2}]$')
axarr[1, 2].set_xlabel(r'$\log \ L_{\rm pri} \ [L_\odot h^{-2}]$')
axarr[1, 3].set_xlabel(r'$\log \ L_{\rm pri} \ [L_\odot h^{-2}]$')

plt.tight_layout(pad=0.6)
plt.subplots_adjust(hspace=0.0)
plt.savefig(DIRECTORY + 'errors.pdf')
plt.savefig(DIRECTORY + 'errors.png', dpi=300)
