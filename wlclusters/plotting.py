import numpy as np
from .WlModel import WLData
from .WlModel import WlModel_np

def wldata_from_ID(
    lens_id,
    cluster_cat,
    shear_profiles,
    results,
    all_chains=None,
    return_shear=False,
    return_shear_model="envelope",
    cosmo=None,
    delta=200
):
    """
    Initialize a WLData class for an individual cluster, optionally computing shear profiles.

    This function creates a `WLData` object based on the given cluster lens ID.
    If done after fitting, it can optionally compute shear profiles under two scenarios:

    - `'envelope'`: Computes the shear profile for each set of parameters in `all_chains`.
    - `'median parameters'`: Computes the shear profile for only the median parameters.

    Parameters
    ----------
    lens_id : int or list of int
        ID of the lens or a list of lens IDs.
    cluster_cat : ndarray or pandas.DataFrame
        Catalog containing cluster information.
    shear_profiles : ndarray or pandas.DataFrame
        Catalog containing shear profile information.
    results : ndarray or pandas.DataFrame
        Results from the lensing analysis.
    all_chains : ndarray, optional
        Posterior chains for parameter estimates (required for the 'envelope' model).
        Defaults to None.
    return_shear : bool, optional
        If True, returns shear profile data. Defaults to False.
    return_shear_model : str, optional
        Type of shear model to return, either 'median parameters' or 'envelope'.
        Defaults to 'envelope'.
    cosmo : object, optional
        Cosmological parameters object. Defaults to None.

    Returns
    -------
    wldata : WLData
        Weak lensing data object.
    tuple, optional
        If `return_shear=True`, a tuple is returned:

        - If `return_shear_model='median parameters'`:
            - `gplus_med`: Shear profile for the median parameters.
            - `rm`: Radial bins for the shear profile.

        - If `return_shear_model='envelope'`:
            - `gplus_results`: Shear profiles for sampled parameter sets from `all_chains`.
            - `rm`: Radial bins for the shear profile.

    Raises
    ------
    ValueError
        If `all_chains` is None when `return_shear_model='envelope'` is selected.
    """

    # Extract relevant data based the ID of the chosen cluster
    results_id = results[results["ID"] == lens_id]
    shear_id = shear_profiles[shear_profiles["ID"] == lens_id]
    z_cl = cluster_cat[cluster_cat["ID"] == lens_id]["z_p"]
    rin = shear_id["rin"]
    rout = shear_id["rout"]
    gplus = shear_id["gplus"]
    err_g = shear_id["errors"]
    msci = shear_id["msci"][0]
    fl = shear_id["fl"][0]

    # Create a WLData object with the extracted data
    wldata = WLData(
        redshift=z_cl,
        rin=rin,
        rout=rout,
        gplus=gplus,
        err_gplus=err_g,
        sigmacrit_inv=msci,
        fl=fl,
        cosmo=cosmo,
        delta=delta
    )

    # Check if shear profiles should be returned
    if return_shear:

        if return_shear_model == "median parameters":

            pmod_med = [results_id["c200_med"], results_id["r200_med"]]
            # Compute shear profile
            model = WlModel_np(wldata, delta=delta)
            gplus_med, rm, ev = model.run(pmod_med)
            # Mask to cut the radial extrapolation
            mask = (rm >= min(wldata.rin_wl)) & (rm <= max(wldata.rout_wl))
            return wldata, gplus_med[mask], rm[mask]

        elif return_shear_model == "envelope":
            # Initialize the WlModel
            model = WlModel_np(wldata, delta=delta)

            if all_chains is None:
                print("Error: posterior chains are needed to compute the envelope.")
                return

            chains = all_chains[np.isin(all_chains["ID"], lens_id)]

            if "log10mdelt" in chains.colnames and "log10cdelt" in chains.colnames:
                chains["mdelt"] = 10 ** chains["log10mdelt"]
                chains["cdelt"] = 10 ** chains["log10cdelt"]

            if "mdelt" in chains.colnames:
                chains['rdelt'] = model.mdelt_to_rdelt(chains['mdelt'])

            # Sample 500 rows randomly from the chains for computational efficiency
            sampled_indices = np.random.choice(np.arange(len(chains['cdelt'][0])), 500, replace=False)
            pmod = np.array(
                (chains['cdelt'][0][sampled_indices], chains['rdelt'][0][sampled_indices])
            ).T
            gplus_results = []

            # Loop through the sampled chain rows to compute shear profile
            for pmod_i in pmod:
                gplus, rm, ev = model.run(pmod_i)
                # Mask to cut the radial extrapolation
                mask = (rm >= min(wldata.rin_wl)) & (rm <= max(wldata.rout_wl))
                gplus_results.append(gplus[mask])

            return wldata, gplus_results, rm[mask]

    return wldata