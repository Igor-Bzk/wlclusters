# Copyright (C) 2012-2026, CEA and contributors (for the Euclid Science Ground
# Segment) SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from scipy.spatial import cKDTree
import healpy as hp
from astropy.table import Table

def healpix2rad(nside, dist):
    return hp.nside2resol(nside) * dist

def matchZ(wl_catalog, z_catalog, scale, col_wl=["RA_OBJ", "DEC_OBJ", "SNR"], col_z=["RA", "DEC", "SNR", "z_p"]):
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
    tree = cKDTree(vecs_z)
    
    z_col = np.full_like(ra_wl, -np.inf, dtype=float)
    
    for clust_id in snr_order:
        dists, matches = tree.query(vecs_wl[clust_id], k=5, distance_upper_bound=max_radius)
        true_matches = matches[dists <= max_radius]
        if len(true_matches) > 0:
            best_match = true_matches[np.argmax(snr_z[true_matches])]
            z_col[clust_id] = snr_z[best_match]
            snr_z[best_match] = -np.inf  # Mark this z_catalog entry as used
    
    matched_mask = np.isfinite(z_col)
    return Table([ra_wl[matched_mask], dec_wl[matched_mask], snr_wl[matched_mask], z_col[matched_mask]],
                 names=col_wl + ['z_p'])