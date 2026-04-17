import pymc as pm
import numpy as np
from astropy.table import Table
from tqdm import tqdm
from .modeling import WLData, WLmodel
from .utils import *
from warnings import warn


def select_covariance(covtype, input_covmat, clust_id, clust_z, cluster_profiles):
    """
    Selects the appropriate covariance matrix based on the type of covariance specified.

    Args:
        covtype (str): The type of covariance matrix to use. Options are 'lss_cov', 'tot_cov', or 'None'.
        input_covmat (Table): The input covariance matrix table containing cluster IDs or redshifts.
        clust_id (int): The ID of the current cluster.
        clust_z (float): The redshift of the current cluster.
        cluster_profiles (Table): Table containing the cluster shear profile information including statistical errors.

    Returns:
        np.ndarray: The selected covariance matrix.
    """
    if covtype == "lss_cov":
        lss_cov = input_covmat
        if "ID" in lss_cov.colnames:
            return lss_cov["covariance_matrix"][np.isin(lss_cov["ID"], clust_id)][0]
        elif "z" in lss_cov.colnames:
            closest_z = find_closest_redshift(clust_z, lss_cov["z"])
            return lss_cov[np.isin(lss_cov["z"], closest_z)]["covariance_matrix"][
                0
            ] + np.diag(np.square(cluster_profiles["errors"]))
    elif covtype == "tot_cov":
        tot_cov = input_covmat
        if "ID" in tot_cov.colnames:
            return tot_cov["covariance_matrix"][np.isin(tot_cov["ID"], clust_id)][0]
        elif "z" in tot_cov.colnames:
            closest_z = find_closest_redshift(clust_z, tot_cov["z"])
            return tot_cov[np.isin(tot_cov["z"], closest_z)]["covariance_matrix"][0]
    else:
        return np.diag(np.square(cluster_profiles["errors"]))


def setup_parameters(parnames, cosmo, clust_z, delta=200):
    """
    Sets up the parameters for the NFW profile model based on the chosen parameterization by converting the user choice into cdelt and rdelt.

    Args:
        parnames (list): List of parameter names to be used in the model (e.g. ['cdelt', 'rdelt'], ['cdelt', 'mdelt'], etc.).
        cosmo (astropy.cosmology.Cosmology): The cosmology object to be used for calculations.
        clust_z (float):The redshift of the current cluster.

    Returns:
        list: List of parameters for the model.
    """
    if parnames == ["cdelt", "rdelt"]:
        cdelt = pm.Uniform(name="cdelt", lower=1.0, upper=10.0)
        rdelt = pm.Uniform(name="rdelt", lower=200.0, upper=4000.0)
        pmod = [cdelt, rdelt]
    elif parnames == ["cdelt", "mdelt"]:
        cdelt = pm.Uniform(name="cdelt", lower=1.0, upper=10.0)
        mdelt = pm.Uniform(name="mdelt", lower=1e12, upper=1e16)
        rdelt = pm.Deterministic("rdelt", mdelt_to_rdelt(mdelt, clust_z, cosmo, delta))
        pmod = [cdelt, rdelt]
    elif parnames == ["log10cdelt", "log10mdelt"]:
        log10cdelt = pm.Uniform(name="log10cdelt", lower=0.0, upper=1.0)
        log10mdelt = pm.Uniform(name="log10mdelt", lower=12.0, upper=16.0)
        cdelt = pm.Deterministic("cdelt", 10**log10cdelt)
        mdelt = pm.Deterministic("mdelt", 10**log10mdelt)
        rdelt = pm.Deterministic("rdelt", mdelt_to_rdelt(mdelt, clust_z, cosmo, delta))
        pmod = [cdelt, rdelt]
    elif parnames == ["cdelt", "log10mdelt"]:
        cdelt = pm.Uniform(name="cdelt", lower=1.0, upper=10.0)
        log10mdelt = pm.Uniform(name="log10mdelt", lower=12.0, upper=16.0)
        mdelt = pm.Deterministic("mdelt", 10**log10mdelt)
        rdelt = pm.Deterministic("rdelt", mdelt_to_rdelt(mdelt, clust_z, cosmo, delta))
        pmod = [cdelt, rdelt]
    else:
        raise ValueError("Invalid parnames specified.")
    return pmod


def forward_model(wldata, parnames, cosmo, clust_z, cov_mat, ndraws, ntune, delta=200):
    """
    Performs forward modeling of weak lensing data using a specified NFW profile and PyMC.

    Args:
        wldata (class WLData): The weak lensing data object.
        parnames (list): List of parameter names (strings) to be used in the model.
        cosmo (astropy.cosmology.Cosmology): The cosmology object to be used for calculations.
        clust_z (float): The redshift of the current cluster.
        cov_mat (np.ndarray): Covariance matrix for the weak lensing data.

    Returns:
        trace : pymc5.backends.base.MultiTrace, the trace of the MCMC sampling process.
    """
    with pm.Model() as model:
        # Setup parameters inside the model context
        pmod = setup_parameters(parnames, cosmo, clust_z, delta)

        # Build the weak lensing model
        gmodel, rm, ev = WLmodel(wldata, pmod, delta=delta)
        g_obs = pm.MvNormal("WL", mu=gmodel[ev], observed=wldata.gplus, cov=cov_mat)

        # Sample the posterior
        trace = pm.sample(draws=ndraws, tune=ntune)

    return trace


def extract_results(cluster_cat, all_chains, unit, cosmo, parnames, delta):
    """
    Extracts the weak lensing modeling results, computing medians and percentiles for mass, radius, and concentration.

    Args:
    cluster_cat (Table): The catalog of clusters with ID and redshift information.
    all_chains (Table): The posterior chains for concentration and radius/mass.
    unit (str): The unit system to use ('proper' or 'comoving').
    cosmo (astropy.cosmology.Cosmology): The cosmology object to be used for calculations.
    parnames (list): List of parameter names used in the model.

    Returns:
        Table: Table containing the extracted results for each cluster (m200, r200, c200).
    """
    z_p = cluster_cat["z_p"]

    delta = str(int(delta))
    
    if parnames == ["cdelt", "rdelt"]:

        c_delta_med = np.median(all_chains["cdelt"], axis=1)
        c_delta_perc_16 = np.percentile(all_chains["cdelt"], 16, axis=1)
        c_delta_perc_84 = np.percentile(all_chains["cdelt"], 84, axis=1)

        r_delta_med = np.median(all_chains["rdelt"], axis=1)
        r_delta_perc_16 = np.percentile(all_chains["rdelt"], 16, axis=1)
        r_delta_perc_84 = np.percentile(all_chains["rdelt"], 84, axis=1)

        if unit == "proper":
            m_delta_med = rdelt_to_mdelt(r_delta_med, z_p, cosmo)
            m_delta_perc_16 = rdelt_to_mdelt(r_delta_perc_16, z_p, cosmo)
            m_delta_perc_84 = rdelt_to_mdelt(r_delta_perc_84, z_p, cosmo)
        elif unit == "comoving":
            r_delta_proper_med = r_delta_med * (1 / (1 + z_p))
            r_delta_proper_perc_16 = r_delta_perc_16 * (1 / (1 + z_p))
            r_delta_proper_perc_84 = r_delta_perc_84 * (1 / (1 + z_p))
            m_delta_med = rdelt_to_mdelt(r_delta_proper_med, z_p, cosmo)
            m_delta_perc_16 = rdelt_to_mdelt(r_delta_proper_perc_16, z_p, cosmo)
            m_delta_perc_84 = rdelt_to_mdelt(r_delta_proper_perc_84, z_p, cosmo)

    elif parnames == ["cdelt", "mdelt"]:

        c_delta_med = np.median(all_chains["cdelt"], axis=1)
        c_delta_perc_16 = np.percentile(all_chains["cdelt"], 16, axis=1)
        c_delta_perc_84 = np.percentile(all_chains["cdelt"], 84, axis=1)

        m_delta_med = np.median(all_chains["mdelt"], axis=1)
        m_delta_perc_16 = np.percentile(all_chains["mdelt"], 16, axis=1)
        m_delta_perc_84 = np.percentile(all_chains["mdelt"], 84, axis=1)

        if unit == "proper":
            r_delta_med = mdelt_to_rdelt(m_delta_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_perc_84, z_p, cosmo)
        elif unit == "comoving":
            m_delta_proper_med = m_delta_med * (1 / (1 + z_p))
            m_delta_proper_perc_16 = m_delta_perc_16 * (1 / (1 + z_p))
            m_delta_proper_perc_84 = m_delta_perc_84 * (1 / (1 + z_p))
            r_delta_med = mdelt_to_rdelt(m_delta_proper_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_proper_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_proper_perc_84, z_p, cosmo)

    elif parnames == ["log10cdelt", "log10mdelt"]:

        c_delta_med = np.median(10 ** all_chains["log10cdelt"], axis=1)
        c_delta_perc_16 = np.percentile(10 ** all_chains["log10cdelt"], 16, axis=1)
        c_delta_perc_84 = np.percentile(10 ** all_chains["log10cdelt"], 84, axis=1)

        m_delta_med = np.median(10 ** all_chains["log10mdelt"], axis=1)
        m_delta_perc_16 = np.percentile(10 ** all_chains["log10mdelt"], 16, axis=1)
        m_delta_perc_84 = np.percentile(10 ** all_chains["log10mdelt"], 84, axis=1)

        if unit == "proper":
            r_delta_med = mdelt_to_rdelt(m_delta_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_perc_84, z_p, cosmo)
        elif unit == "comoving":
            m_delta_proper_med = m_delta_med * (1 / (1 + z_p))
            m_delta_proper_perc_16 = m_delta_perc_16 * (1 / (1 + z_p))
            m_delta_proper_perc_84 = m_delta_perc_84 * (1 / (1 + z_p))
            r_delta_med = mdelt_to_rdelt(m_delta_proper_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_proper_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_proper_perc_84, z_p, cosmo)

    elif parnames == ["cdelt", "log10mdelt"]:

        c_delta_med = np.median(all_chains["cdelt"], axis=1)
        c_delta_perc_16 = np.percentile(all_chains["cdelt"], 16, axis=1)
        c_delta_perc_84 = np.percentile(all_chains["cdelt"], 84, axis=1)

        m_delta_med = np.median(10 ** all_chains["log10mdelt"], axis=1)
        m_delta_perc_16 = np.percentile(10 ** all_chains["log10mdelt"], 16, axis=1)
        m_delta_perc_84 = np.percentile(10 ** all_chains["log10mdelt"], 84, axis=1)

        if unit == "proper":
            r_delta_med = mdelt_to_rdelt(m_delta_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_perc_84, z_p, cosmo)
        elif unit == "comoving":
            m_delta_proper_med = m_delta_med * (1 / (1 + z_p))
            m_delta_proper_perc_16 = m_delta_perc_16 * (1 / (1 + z_p))
            m_delta_proper_perc_84 = m_delta_perc_84 * (1 / (1 + z_p))
            r_delta_med = mdelt_to_rdelt(m_delta_proper_med, z_p, cosmo)
            r_delta_perc_16 = mdelt_to_rdelt(m_delta_proper_perc_16, z_p, cosmo)
            r_delta_perc_84 = mdelt_to_rdelt(m_delta_proper_perc_84, z_p, cosmo)

    if unit == "proper":
        m_delta_med = rdelt_to_mdelt(r_delta_med, z_p, cosmo)
        m_delta_perc_16 = rdelt_to_mdelt(r_delta_perc_16, z_p, cosmo)
        m_delta_perc_84 = rdelt_to_mdelt(r_delta_perc_84, z_p, cosmo)
    elif unit == "comoving":
        r_delta_proper_med = r_delta_med * (1 / (1 + z_p))
        r_delta_proper_perc_16 = r_delta_perc_16 * (1 / (1 + z_p))
        r_delta_proper_perc_84 = r_delta_perc_84 * (1 / (1 + z_p))
        m_delta_med = rdelt_to_mdelt(r_delta_proper_med, z_p, cosmo)
        m_delta_perc_16 = rdelt_to_mdelt(r_delta_proper_perc_16, z_p, cosmo)
        m_delta_perc_84 = rdelt_to_mdelt(r_delta_proper_perc_84, z_p, cosmo)

    results_table = Table()
    results_table["ID"] = cluster_cat["ID"]
    results_table[f"m{delta}_med"] = m_delta_med
    results_table[f"m{delta}_perc_16"] = m_delta_perc_16
    results_table[f"m{delta}_perc_84"] = m_delta_perc_84
    results_table[f"r{delta}_med"] = r_delta_med
    results_table[f"r{delta}_perc_16"] = r_delta_perc_16
    results_table[f"r{delta}_perc_84"] = r_delta_perc_84
    results_table[f"c{delta}_med"] = c_delta_med
    results_table[f"c{delta}_perc_16"] = c_delta_perc_16
    results_table[f"c{delta}_perc_84"] = c_delta_perc_84

    return results_table


def run(
    cluster_cat,
    shear_profiles,
    cosmo,
    covtype="None",
    input_covmat=None,
    unit="proper",
    ndraws=2000,
    ntune=1000,
    parnames=["cdelt", "rdelt"],
    delta=200,
):
    """
    Executes the full weak lensing modeling pipeline for a catalog of clusters.

    Args:
        cluster_cat (Table): The catalog of clusters with ID and redshift information.
        shear_profiles (Table): The shear profiles for each cluster, containing gplus and error data.
        cosmo (astropy.cosmology.Cosmology): The cosmology object to be used for calculations.
        covtype (str): optional, the type of covariance matrix to use ('lss_cov', 'tot_cov', or 'None'). Default is 'None'.
        input_covmat (Table): optional, input covariance matrix table if applicable.
        unit (str): optional, unit system to use ('proper' or 'comoving'). Default is 'proper'.
        ndraws (int): optional, number of draws for MCMC sampling. Default is 2000.
        ntune (int): optional, number of tuning steps for MCMC. Default is 1000.
        parnames (list): optional, list of parameter names used for modeling. Default is ['cdelt', 'rdelt'].

    Returns:
        Table: Table containing the posterior chains and the extracted results for each cluster.
    """
    all_c200_chains = []
    all_r200_chains = []
    valid_cluster_ids = []

    for cluster in tqdm(cluster_cat):
        clust_id = cluster["ID"]
        if clust_id not in shear_profiles["ID"]:
            warn(f"Warning: Cluster ID {clust_id} not found in shear profiles. Skipping this cluster.")
            continue
        clust_z = cluster["z_p"]

        mask = shear_profiles["ID"] == clust_id
        cluster_profiles = shear_profiles[mask]

        cov_mat = select_covariance(
            covtype, input_covmat, clust_id, clust_z, cluster_profiles
        )

        rin = cluster_profiles["rin"]
        rout = cluster_profiles["rout"]
        gplus = cluster_profiles["gplus"]
        errors = cluster_profiles["errors"]

        mean_sigcrit_inv, fl = cluster_profiles["msci"][0], cluster_profiles["fl"][0]

        wldata = WLData(
            redshift=clust_z,
            rin=rin,
            rout=rout,
            gplus=gplus,
            err_gplus=errors,
            sigmacrit_inv=mean_sigcrit_inv,
            fl=fl,
            cosmo=cosmo,
            unit=unit,
            delta=delta,
        )

        # Call forward_model with all arguments
        try:
            trace = forward_model(wldata, parnames, cosmo, clust_z, cov_mat, ndraws, ntune, delta=delta)
        except Exception as e:
            warn(f"Error processing cluster ID {clust_id}:\n {e}\n Skipping this cluster.")
            continue
        
        valid_cluster_ids.append(clust_id)
        all_c200_chains.append(np.array(trace.posterior[parnames[0]]).flatten())
        all_r200_chains.append(np.array(trace.posterior[parnames[1]]).flatten())

    all_chains = Table()
    all_chains["ID"] = valid_cluster_ids
    all_chains[str(parnames[0])] = all_c200_chains
    all_chains[str(parnames[1])] = all_r200_chains
    
    if len(all_chains) == 0:
        warn("No valid clusters were processed. Returning an empty table.")
        return Table()  # Return an empty table if no valid clusters were processed

    return all_chains, extract_results(cluster_cat[np.isin(cluster_cat["ID"], valid_cluster_ids)], all_chains, unit, cosmo, parnames, delta)
