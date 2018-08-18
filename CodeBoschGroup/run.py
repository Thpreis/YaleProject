import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pf
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM, Planck13
from scipy import constants
import pymangle
from satellite_kinematics import z_min, z_max, sigma_200
from satellite_kinematics import get_fiber_collision_probability
from satellite_kinematics import write_catalog_to_file
import matplotlib

catalog_post = pf.getdata("post_catalog.dr72bright0.fits")
kcorrect = pf.getdata("kcorrect.nearest.petro.z0.10.fits")
catalog_pre = pf.getdata("lss_index.dr72.fits")
mask_info = pf.getdata("lss_combmask.dr72.fits")
npts = len(catalog_post)

print("Total number of data points: %d" % npts)
print("Number given in Zehavi+11: ~570,000")
print("Apparent magnitude limit: %.2f" % np.max(catalog_post['M']))
print("Number of galaxies with r < 14.5: %d" %
      np.sum(catalog_post['M'] < 14.5))
print("Number given in Zehavi et al. (2011): ~6,000")

print("Total usable area: %.1f sq degree" % (
    np.sum(mask_info['STR'][mask_info['FGOTMAIN'] >= 0.8]) *
    (180.0 / np.pi)**2))

is_collided = catalog_pre['Z'][catalog_post['OBJECT_POSITION']] == -1
print("Spectroscopic completeness: %.2f%%" % (
    100 * (1 - np.mean(is_collided))))
icomb = catalog_pre['ICOMB'][catalog_post['OBJECT_POSITION']]

# %% Plot the listed absolute magnitudes against manually calculated ones.
cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Tcmb0=0, Neff=0)
m_abs_1 = kcorrect['ABSMAG'][:, 2][catalog_post['OBJECT_POSITION']]
distmod = cosmo.distmod(catalog_post['Z']).value
k = kcorrect['KCORRECT'][:, 2][catalog_post['OBJECT_POSITION']]
m_abs_2 = catalog_post['M'] - k - distmod + 0.010

plt.scatter(catalog_post['Z'][::30], m_abs_1[::30] - m_abs_2[::30],
            color='black', s=1, edgecolors='none')
plt.xlim(0.0, 0.4)
plt.xlabel("Redshift")
plt.ylabel(r"$M_{r, \mathrm{catalog}} - M_{r, \mathrm{calculated}}$")
plt.tight_layout(pad=0.25)
plt.savefig("m_calculated_vs_catalog.pdf")
plt.close()

# In the following, I will use absolute magnitudes calculated assuming Planck13
# cosmological parameters.
distmod = Planck13.distmod(catalog_post['Z']).value
k = kcorrect['KCORRECT'][:, 2][catalog_post['OBJECT_POSITION']]
m_abs = (catalog_post['M'] - k - distmod - 5 *
         np.log10(Planck13.H0.value / 100.0) + 0.010)

# We also apply the evolution correction. See
# http://cosmo.nyu.edu/blanton/vagc/lss.html for details.
q0 = 2.0
q1 = -1.0
m_abs = (m_abs + q0 * (1 + q1 * (catalog_post['Z'] - 0.1)) *
         (catalog_post['Z'] - 0.1))

# Let's compare some numbers with Zehavi et al. (2011).
mask = ((m_abs <= -19.5) & (0.02 <= catalog_post['Z']) &
        (catalog_post['Z'] <= 2.545e7 / constants.c))
print("Galaxies with M_r < -19.5 and zc < 25,450")
print(("Number in this catalog: %d" % np.sum(mask)))
print("Number quoted in Zehavi+11: 132664")

# Calculate luminosities.
lum = 10**(-0.4 * (m_abs - 4.76))
plt.scatter(catalog_post['Z'][::10], np.log10(lum[::10]), color='black', s=1,
            edgecolors='none')
plt.plot([z_min, z_min, z_max, z_max], [12.0, 9.5, 9.5, 12.0])
plt.xlim(0.01, 0.25)
plt.ylim(8, 12)
plt.xlabel(r"$z$")
plt.ylabel(r"$\log L \ [L_\odot h^{-2}]$")
plt.tight_layout(pad=0.25)
plt.savefig('lum_vs_z.pdf')
plt.savefig('lum_vs_z.png', dpi=300)
plt.close()

print('Number of primary candidates: %d' % np.sum(
      (z_min < catalog_post['Z']) & (catalog_post['Z'] < z_max) &
      (lum > 10**(9.5)) & ~is_collided))


# %% Now, we calculate colors.

m_abs_g = kcorrect['ABSMAG'][:, 1][catalog_post['OBJECT_POSITION']]
m_abs_r = kcorrect['ABSMAG'][:, 2][catalog_post['OBJECT_POSITION']]
color = np.where(m_abs_g - m_abs_r > 0.21 - 0.03 * m_abs, 'red', 'blue')

mask = ((np.log10(lum) >= 9.5) & (z_min <= catalog_post['Z']) &
        (catalog_post['Z'] < z_max) & (color == 'red'))
plt.scatter(m_abs[mask][::10], (m_abs_g - m_abs_r)[mask][::10],
            0.1, color='red')

mask = ((np.log10(lum) >= 9.5) & (z_min <= catalog_post['Z']) &
        (catalog_post['Z'] < z_max) & (color == 'blue'))
plt.scatter(m_abs[mask][::10], (m_abs_g - m_abs_r)[mask][::10],
            0.1, color='blue')

plt.xlabel(r"$\log L \ [h^{-2} L_\odot]$")
plt.ylabel(r"Color $g - r$")
plt.ylim(0.3, 1.1)
plt.tight_layout(pad=0.25)
plt.savefig('color_vs_lum.pdf')
plt.close()


# %% Let's do a fun illustrative plot.
cmap = matplotlib.cm.get_cmap('jet')
mask = (np.abs(catalog_post['DEC']) < 1.25) & (catalog_post['Z'] < 0.3)
plt.figure(figsize=(4, 4))
ax = plt.subplot(111, projection='polar')
ax.scatter(catalog_post['RA'][mask] * np.pi / 180, catalog_post['Z'][mask],
           c=(catalog_post['Z'][mask] / 0.3), s=1, edgecolors='none')
ax.set_rmax(0.3)
ax.set_rticks([0.05, 0.10, 0.15, 0.20, 0.25])
ax.set_rlabel_position(90)
plt.tight_layout(pad=0.25)
plt.savefig('wedge.pdf')
plt.close()

# %% We now turn to the geometry and mangle.
combmask = pymangle.Mangle("lss_combmask_pix.dr72.ply")
window = pymangle.Mangle("window_pix.dr72bright0.ply")
mask = pymangle.Mangle("mask_pix.dr72bright0.ply")

# %% Almost all galaxies should be within the window function of the respective
# sample. A few could fall off due to round-off errors.
in_window = window.contains(catalog_post['RA'], catalog_post['DEC'])
print("Inside window: %d/%d" % (np.sum(in_window), len(in_window)))

# %% Almost no galaxy should lie inside the mask.
in_mask = mask.contains(catalog_post['RA'], catalog_post['DEC'])
print("Inside mask: %d/%d" % (np.sum(in_mask), len(in_mask)))

# %% The polygon ID provided in the catalog should also be the same then the
# one we get out of the mask file for each galaxy. Small differences are
# probably due to overlapping polygons.
polyid = combmask.polyid(catalog_post['RA'], catalog_post['DEC'])
print("Consistent poly ID: %d/%d" % (np.sum(icomb == polyid), len(icomb)))

# %% Let's make a plot of the geometry.
ra_grid = np.linspace(0, 360, 500)
dec_grid = np.linspace(-20, 90, 500)
ra_grid, dec_grid = np.meshgrid(ra_grid, dec_grid)

in_window_grid = np.reshape(window.contains(
    np.ravel(ra_grid), np.ravel(dec_grid)), ra_grid.shape)

in_mask_grid = np.reshape(mask.contains(
    np.ravel(ra_grid), np.ravel(dec_grid)), ra_grid.shape)

polyid_grid = np.reshape(combmask.polyid(np.ravel(ra_grid),
                                         np.ravel(dec_grid)),
                         ra_grid.shape)
fgotmain_grid = mask_info['FGOTMAIN'][polyid_grid]
fgotmain_grid = np.where((polyid_grid != -1) & (fgotmain_grid >= 0.8),
                         fgotmain_grid, np.nan)
fgotmain_grid = np.where(in_window_grid, fgotmain_grid, np.nan)
fgotmain_grid = np.where(np.logical_not(in_mask_grid), fgotmain_grid, np.nan)

plt.figure(figsize=(7, 4))
plt.contourf(ra_grid, dec_grid, fgotmain_grid)
sel = mask_info['FGOTMAIN'][icomb] >= 0.8
random = np.arange(np.sum(sel))
np.random.shuffle(random)
cb = plt.colorbar()
cb.set_label(r"FGOTMAIN")
plt.scatter(catalog_post['RA'][sel][random][::30],
            catalog_post['DEC'][sel][random][::30], color='black', s=1,
            edgecolors='none')
plt.xlim(0, 360)
plt.ylim(-20, 75)
plt.xlabel(r"Right Ascension $\alpha / \mathrm{deg}$")
plt.ylabel(r"Declination $\delta / \mathrm{deg}$")
plt.tight_layout(pad=0.3)
plt.savefig('geometry.pdf')
plt.close()

# Calculate completeness scores as explained in the draft.
d_com = (Planck13.comoving_distance(catalog_post['Z']).value *
         Planck13.H0.value / 100.0)
s_200 = sigma_200(lum, color)
r_h = 0.8 * s_200 / d_com * 180 / np.pi
plt.hist(np.log10(r_h), bins=np.linspace(-1.1, 0.2, 100), histtype='step',
         normed=True)
plt.xlabel(r'Radius $\log r_h / \mathrm{deg}$')
plt.ylabel(r'Distribution')
plt.tight_layout(pad=0.25)
plt.savefig('cylinder_sizes.pdf')
plt.close()

X_BINS = np.linspace(-1, 1, 20)
Y_BINS = np.linspace(-1, 1, 20)

completeness = np.zeros(len(catalog_post['RA']))
weight_tot = 0

for x in X_BINS:
    for y in Y_BINS:
        if x**2 + y**2 <= 1:
            ra = catalog_post['RA'] + x * r_h / np.cos(catalog_post['DEC'] *
                                                       np.pi / 180.0)
            dec = catalog_post['DEC'] + y * r_h
            completeness += (window.contains(ra, dec) &
                             np.logical_not(mask.contains(ra, dec)))
            weight_tot += 1

completeness /= weight_tot

plt.figure(figsize=(7, 4))
random = np.arange(np.sum(sel))
np.random.shuffle(random)
cb.set_label(r"Completeness")
plt.scatter(catalog_post['RA'][sel][random][::5],
            catalog_post['DEC'][sel][random][::5],
            c=completeness[sel][random][::5], s=1, edgecolors='none')
cb = plt.colorbar()
cb.set_label(r"Completeness")
plt.xlim(100, 250)
plt.ylim(-10, 20)
plt.xlabel(r"Right Ascension $\alpha / \mathrm{deg}$")
plt.ylabel(r"Declination $\delta / \mathrm{deg}$")
plt.tight_layout(pad=0.3)
plt.savefig('completeness.pdf')

# %% Makes the catalog.

catalog = Table()

catalog['gal_id'] = catalog_post['OBJECT_POSITION']
catalog['z_spec'] = catalog_post['Z']
catalog['luminosity'] = lum
catalog['collision'] = is_collided
catalog['ra'] = catalog_post['RA']
catalog['dec'] = catalog_post['DEC']
catalog['d_com'] = d_com
catalog['f_spec'] = mask_info['FGOTMAIN'][icomb]
catalog['completeness'] = completeness
catalog['color'] = color
print('Probability to be collided if you have a close neighbour: %.1f%%' % (
    100 * np.sort(np.unique(1.0 -
                            get_fiber_collision_probability(catalog)))[1]))
catalog['weight_spec'] = 1.0 / get_fiber_collision_probability(catalog)

# %% Writes the catalog.

mask = (catalog['luminosity'] > 10**9.5)
print("Number of galaxies passing cuts: %d" % np.sum(mask))
print("Number of fiber collisions: %d" % np.sum(is_collided[mask]))

write_catalog_to_file(catalog[mask], 'sdss.hdf5')

