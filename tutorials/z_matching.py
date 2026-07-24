# Copyright (C) 2012-2026, CEA and contributors (for the Euclid Science Ground
# Segment) SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from scipy.spatial import cKDTree
import healpy as hp
from astropy.table import Table

from colossus.halo import profile_nfw, concentration
from colossus.cosmology import cosmology
from astropy.cosmology import FlatLambdaCDM
from astropy import constants
from astropy import units as u

def healpix2rad(nside, dist):
    return hp.nside2resol(nside) * dist

def matchZ(wl_catalog, z_catalog, scale, col_wl=["RA_OBJ", "DEC_OBJ", "SNR"], col_z=["RA", "DEC", "SNR", "z_p"]):
    """
    Match weak lensing detections with photometric redshifts from a different catalog.
    With the method described in Chappuis et al. 2026.
    
    Args:
        wl_catalog (Table): weak lensing catalog with RA, Dec, and SNR columns.
        z_catalog (Table): catalog with RA, Dec, SNR, and photometric redshift columns.
        scale (float): detection scale / max_radius in radians.
        col_wl (list): list of column names in wl_catalog : Ra, Dec, SNR. Defaults to ["RA_OBJ", "DEC_OBJ", "SNR"].
        col_z (list): list of column names in z_catalog : Ra, Dec, SNR, z_p. Defaults to ["RA", "DEC", "SNR", "z_p"].
    Returns:
        Redshift array of the matched clusters. Set to NaN for unmatched clusters.
    """
    max_radius = 2 * np.sin(scale / 2)
    snr_wl = wl_catalog[col_wl[2]]
    snr_order = np.argsort(snr_wl)
    ra_wl = wl_catalog[col_wl[0]]
    dec_wl = wl_catalog[col_wl[1]]
    vecs_wl = np.asarray(hp.ang2vec(ra_wl, dec_wl, lonlat=True))
    
    ra_z = z_catalog[col_z[0]]
    dec_z = z_catalog[col_z[1]]
    vecs_z = np.asarray(hp.ang2vec(ra_z, dec_z, lonlat=True))
    snr_z = z_catalog[col_z[2]].copy()
    zs = z_catalog[col_z[3]]
    tree = cKDTree(vecs_z)
    
    z_col = np.full_like(ra_wl, np.nan, dtype=float)
    
    dists, matches = tree.query(vecs_wl, k=5, distance_upper_bound=max_radius)
    for clust_id in snr_order:
        true_matches = matches[clust_id][dists[clust_id] <= max_radius]
        if len(true_matches) > 0:
            best_match = true_matches[np.argmax(snr_z[true_matches])]
            z_col[clust_id] = zs[best_match]
            snr_z[best_match] = -np.inf  # Mark this z_catalog entry as used

    return z_col

# Define the parameters
Omega_M = 0.31345
Omega_b = 0.0481
Omega_Lambda = 0.68655
h = 0.6731
sigma8 = 0.847

# Calculate Omega_CDM
Omega_CDM = Omega_M - Omega_b

# Define the cosmology
my_cosmo = {'flat': True, 'H0': h*100, 'Om0': Omega_M, 'Ob0': Omega_b, 'sigma8': sigma8, 'ns':0.97}
cosmo_colossus = cosmology.setCosmology('my_cosmo', **my_cosmo)
cosmo = FlatLambdaCDM(H0=h*100, Om0=Omega_M, Ob0=Omega_b)

CONSTE = ((constants.c)**2 / (4*np.pi*constants.G)).to(u.M_sun/u.kpc)

def get_Gc(c):
    """"function of the concentration, Berge+2010 eq. B.7"""
    return(0.131/c**2 - 0.375/c + 0.388 -5*1e-4*c - 2.8*1e-7*c**2)

def get_pz_binning(sources_z_cat, n_bins=50):
    """Bin source redshifts for p(z). Call once per source catalog and
    reuse the result across clusters instead of rebinning every call."""

    bin_edges_input = np.linspace(0, np.max(sources_z_cat), n_bins)
    p_of_z_sources, bin_edges = np.histogram(sources_z_cat, bins=bin_edges_input, density=True)
    z_sources_bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return z_sources_bins, p_of_z_sources

def get_Z_integrated(clust_Z, z_sources_bins, p_of_z_sources):
    """Compute ⟨Z⟩ by integrating over p(z_sources)

    Args:
        z_sources_bins: bin centers of source redshift z_sources
        p_of_z_sources: p(z_sources), the probability density at each z_sources bin

    Returns:
        ⟨Z⟩, the average lensing efficiency for the cluster at clust_Z, integrated over the source redshift distribution.
    """
    Z_values = np.zeros_like(z_sources_bins)

    first_background_index = np.searchsorted(z_sources_bins, clust_Z, side='right')
    if first_background_index < len(z_sources_bins):
        z_sources_background = z_sources_bins[first_background_index:]
        distance_sources = cosmo.angular_diameter_distance(z_sources_background).value
        distance_lens_sources = cosmo.angular_diameter_distance_z1z2(clust_Z, z_sources_background).value
        Z_values[first_background_index:] = distance_lens_sources / distance_sources

    return np.trapezoid(Z_values * p_of_z_sources, z_sources_bins)

def estimateSNR(clust_Z, clust_M, sources_z_cat, z_sources_bins, p_of_z_sources, field_size_deg=10, delta=200):

    """
    Estimate the weak lensing SNR of a given cluster based on its redshift and mass, using the source redshift distribution.
    The initial formula of Berge+2010 initially contains a D_l * Sig_crit_inifity term
    D_l * Sig_crit_infinity = conste * D_l * D_infinity / (D_l * D_infinity) = conste

    Args:
        clust_Z: redshift of the cluster
        clust_M: mass of the cluster (M200)
        sources_z_cat: array of source redshifts
        z_sources_bins, p_of_z_sources: output of get_pz_binning(sources_z_cat)
        field_size_deg: size of the field in degrees (default 10)
        delta: overdensity parameter for mass definition (default 200)
    
    Returns:
        SNR: estimated signal-to-noise ratio for the cluster given its redshift and mass.
    """

    c = concentration.concentration(clust_M*cosmo.h, f'{delta}c', clust_Z, model='bullock01')

    Z = get_Z_integrated(clust_Z, z_sources_bins, p_of_z_sources)

    rho_s, r_s = profile_nfw.NFWProfile.nativeParameters(clust_M*cosmo.h, c, clust_Z, f'{delta}c')

    n_g = (len(sources_z_cat) / (field_size_deg**2)) * 180**2 / (np.pi)**2  # per steradian
    sigma_e = 0.26

    rho_s *= u.M_sun / u.kpc**3
    r_s *= u.kpc

    mu = (
        2 * Z * np.sqrt(2 * np.pi) / sigma_e *
        np.sqrt(n_g) * np.sqrt(get_Gc(c)) *
        rho_s * r_s**2 / CONSTE
    )
    return mu

def estimateCatSNR(clust_z, clust_M, z_sources, field_size_deg=10, n_bins=50, delta=200):
    """Estimate SNR for a catalog of clusters given their redshifts and masses, using the source redshift distribution.

    Args:
        cluster_z: array of cluster redshifts
        cluster_M: array of cluster masses
        z_sources: array of source redshifts
        field_size_deg: size of the field in degrees (default 10)
        n_bins: number of bins for p(z) (default 50)
        delta: overdensity parameter for mass definition (default 200)
    
    Returns:
        snr: array of estimated SNR values for each cluster.
    """

    z_sources_bins, p_of_z_sources = get_pz_binning(z_sources, n_bins)

    snr = np.zeros(len(clust_z))
    z_source_max = np.max(z_sources)
    for i, (z, m) in enumerate(zip(clust_z, clust_M)):
        if z_source_max < z:
            snr[i] = 0.0
        else:
            snr[i] = estimateSNR(z, m, z_sources, z_sources_bins, p_of_z_sources, field_size_deg, delta)
    return snr