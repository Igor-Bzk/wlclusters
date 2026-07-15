import pymc as pm
import numpy as np
from astropy.table import Table
from tqdm import tqdm
from .WlData import WLData
from .WlModel_pymc import WlModel_pymc
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
    if covtype in {"lss_cov", "tot_cov"}:
        if "ID" in input_covmat.colnames:
            return input_covmat["covariance_matrix"][np.isin(input_covmat["ID"], clust_id)][0]
        elif "z" in input_covmat.colnames:
            closest_z = find_closest_redshift(clust_z, input_covmat["z"])
            cov_mat = input_covmat[np.isin(input_covmat["z"], closest_z)]["covariance_matrix"][0]
            if covtype == "lss_cov":
                cov_mat += np.diag(np.square(cluster_profiles["errors"]))
            return cov_mat
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
    match parnames[0]:
        case "cdelt":
            cdelt = pm.Uniform(name="cdelt", lower=1.0, upper=10.0)
        case "log10cdelt":
            cdelt = pm.Uniform(name="log10cdelt", lower=0.0, upper=1.0)
        case _:
            raise ValueError("Invalid parnames specified.")

    match parnames[1]:
        case "rdelt":
            rdelt = pm.Uniform(name="rdelt", lower=200.0, upper=4000.0)
        case "mdelt":
            mdelt = pm.Uniform(name="mdelt", lower=1e12, upper=1e16)
            rdelt = pm.Deterministic("rdelt", mdelt_to_rdelt(mdelt, clust_z, cosmo, delta))
        case "log10mdelt":
            log10mdelt = pm.Uniform(name="log10mdelt", lower=12.0, upper=16.0)
            mdelt = pm.Deterministic("mdelt", 10**log10mdelt)
            rdelt = pm.Deterministic("rdelt", mdelt_to_rdelt(mdelt, clust_z, cosmo, delta))
        case _:
            raise ValueError("Invalid parnames specified.")
    return [cdelt, rdelt]


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
    wlmodel = WlModel_pymc(wldata, parnames, delta=delta)  # Initialize the model to set up necessary attributes
    
    with pm.Model() as model:
        # Setup parameters inside the model context
        # Build the weak lensing model
        gmodel, rm, ev = wlmodel.run()
        pm.MvNormal("WL", mu=gmodel[ev], observed=wldata.gplus, cov=cov_mat)

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
    def delta_stats(key):
        arr = all_chains[key]
        if key.startswith("log10"):
            arr = 10 ** arr
        return list(np.percentile(arr, [16, 50, 84], axis=1))

    results_table = Table([cluster_cat["ID"]], names=["ID"])

    results_table.add_columns((delta_stats(parnames[0])),
        names=[f"c{delta}_perc_16", f"c{delta}_med", f"c{delta}_perc_84"])

    if "rdelt" in parnames[1]:
        r_stats = delta_stats(parnames[1])
        m_colnames = [f"m{delta}_perc_16", f"m{delta}_med", f"m{delta}_perc_84"]
        
        if unit != "proper":
            r_stats /= (1 + z_p)
        results_table.add_columns(rdelt_to_mdelt(r_stats, z_p, cosmo), names=m_colnames)
    else:
        m_stats = delta_stats(parnames[1])
        r_colnames = [f"r{delta}_perc_16", f"r{delta}_med", f"r{delta}_perc_84"]

        if unit != "proper":
            m_stats /= (1 + z_p)
        results_table.add_columns(mdelt_to_rdelt(m_stats, z_p, cosmo), names=r_colnames)

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
    all_chains = Table(names=["ID", str(parnames[0]), str(parnames[1])],
                       dtype=[int, (float, ndraws*2), (float, ndraws*2)])

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
        
        c200_samples = trace.posterior[parnames[0]].values.flatten()
        r200_samples = trace.posterior[parnames[1]].values.flatten()
        
        all_chains.add_row([clust_id, c200_samples, r200_samples])

    if len(all_chains) == 0:
        warn("No valid clusters were processed. Returning an empty table.")
        return Table()  # Return an empty table if no valid clusters were processed

    cluster_cat = cluster_cat[np.isin(cluster_cat["ID"], all_chains["ID"])]

    return all_chains, extract_results(cluster_cat, all_chains, unit, cosmo, parnames, delta)
