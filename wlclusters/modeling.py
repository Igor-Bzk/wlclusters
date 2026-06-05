import numpy as np
import astropy.units as u
from .deproject import deproj_vol
import pymc as pm


def rho_nfw_cr(rmeans, pmod, delta=200.0):
    """
    Computes the Navarro-Frenk-White (NFW) density profile using PyMC (Theano) for a given radial distance array.
    Multiply the result by the critical density of the universe to get the physical density.

    Args:
        radii (array): Radial distances in Mpc.
        pmod (list): Parameters model, containing concentration and radius/mass.
        delta (float, optional): Overdensity parameter, defaults to 200.

    Returns:
        array: NFW density profile divided by the critical density of the universe.
    """

    # Calculate r as the midpoints of radii
    r = rmeans * 1000.0  # Convert radii to kpc

    cdelt, rdelt = pmod

    # Calculate delta_crit using PyMC math functions
    delta_crit = (
        (delta / 3)
        * (cdelt**3)
        * (pm.math.log(1.0 + cdelt) - cdelt / (1 + cdelt)) ** (-1)
    )

    # Return NFW density profile
    return delta_crit / ((cdelt * r / rdelt) * ((1.0 + (cdelt * r / rdelt)) ** 2))


def rho_nfw_cr_np(radii, pmod, delta=200.0):
    """
    Computes the Navarro-Frenk-White (NFW) density profile using NumPy for a given radial distance array.
    Multiply the result by the critical density of the universe to get the physical density.

    Args:
        radii (array): Radial distances in Mpc.
        pmod (list): Parameters model, containing concentration and radius/mass.
        delta (float, optional): Overdensity parameter, defaults to 200.

    Returns:
        array: NFW density profile divided by the critical density of the universe.
    """
    cdelt, rdelt = pmod
    r = (radii[1:] + radii[:-1]) / 2 * 1000.0
    delta_crit = (
        (delta / 3) * (cdelt**3) * (np.log(1.0 + cdelt) - cdelt / (1 + cdelt)) ** (-1)
    )
    return delta_crit / ((cdelt * r / rdelt) * ((1.0 + (cdelt * r / rdelt)) ** 2))


def rho_to_sigma(radii_bins, rho):
    """
    Projects a 3D density profile to compute the surface mass density using PyMC (Theano).

    Args:
        radii_bins (array): Binned radial distances.
        rho (array): 3D density profile values.

    Returns:
        array: Projected surface mass density in units of M_sun * Mpc**-2.
    """

    proj_vol = deproj_vol(radii_bins[:-1], radii_bins[1:])
    area_proj = np.pi * (-((radii_bins[:-1] * 1e6) ** 2) + (radii_bins[1:] * 1e6) ** 2)
    sigma = pm.math.dot(proj_vol, rho) / area_proj
    return sigma * 1e12


def rho_to_sigma_np(radii_bins, rho):
    """
    Projects a 3D density profile to compute the surface mass density using NumPy.

    Args:
        radii_bins (array): Binned radial distances.
        rho (array): 3D density profile values.

    Returns:
        array: Projected surface mass density in units of M_sun * Mpc**-2.
    """

    proj_vol = deproj_vol(radii_bins[:-1], radii_bins[1:])
    area_proj = np.pi * (-((radii_bins[:-1] * 1e6) ** 2) + (radii_bins[1:] * 1e6) ** 2)
    sigma = np.dot(proj_vol, rho) / area_proj
    return sigma * 1e12


def dsigma_trap(sigma, rmean):
    """
    Computes Delta Sigma, the differential surface mass density, using numerical trapezoidal integration.

    Args:
        sigma (array): Projected surface mass density values.
        radii (array): Radial distances.

    Returns:
        array: Differential surface mass density (Delta Sigma).
    """

    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    dr = rmean[1:] - rmean[:-1]

    arg0 = sigma[0] * (rmean2[0] ** 2) / 2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    arg = pm.math.concatenate([[arg0], arg1])
    a = pm.math.cumsum(arg)
    sigmabar = (2 / (rmean**2)) * a
    dsigma = sigmabar - sigma
    return dsigma


def dsigma_trap_np(sigma, radii):
    """
    Computes Delta Sigma using numerical trapezoidal integration with NumPy.

    Args:
        sigma (array): Projected surface mass density values.
        radii (array): Radial distances.

    Returns:
        array: Differential surface mass density (Delta Sigma).
    """
    rmean = (radii[1:] + radii[:-1]) / 2
    rmean2 = (rmean[1:] + rmean[:-1]) / 2
    dr = rmean[1:] - rmean[:-1]

    arg0 = sigma[0] * (rmean2[0] ** 2) / 2
    arg1 = dr * (sigma[1:] * rmean[1:] + sigma[:-1] * rmean[:-1]) / 2

    arg = np.append(arg0, arg1)
    a = np.cumsum(arg)
    sigmabar = (2 / (rmean**2)) * a
    dsigma = sigmabar - sigma
    return dsigma


def get_shear(sigma, dsigma, mean_sigm_crit_inv, fl):
    """
    Computes the expected tangential shear profile using the Seitz and Schneider 1997 (or Umetsu 2020) formula.

    Args:
        sigma (array): Projected surface mass density.
        dsigma (array): Differential surface mass density (Delta Sigma).
        mean_sigm_crit_inv (float): Mean inverse critical surface mass density.
        fl (float): Correction factor for second-order lensing effects.

    Returns:
        array: Mean tangential shear profile.
    """

    shear = (dsigma * mean_sigm_crit_inv) / (1 - fl * sigma * mean_sigm_crit_inv)

    return shear


def get_radplus(radii, rmin=1e-3, rmax=1e2, nptplus=19):
    """
    Generates additional interpolated/extrapolated radii points for integration.

    Args:
        radii (array): Input radii.
        rmin (float, optional): Minimum radius value for extrapolation, defaults to 1e-3.
        rmax (float, optional): Maximum radius value for extrapolation, defaults to 1e2.
        nptplus (int, optional): Number of additional points, defaults to 19.

    Returns:
        tuple:
            - radplus (array): Extended radii array.
            - rmeanplus (array): Midpoint of extended radii.
            - evalrad (array): Indices of original radii within extended radii array.
    """

    if nptplus % 2 == 0:
        nptplus = nptplus + 1
    rmean = (radii[1:] + radii[:-1]) / 2.0

    radstart = np.logspace(np.log10(rmin), np.log10(radii[0]), nptplus)
    radmid = np.linspace(radii[:-1], radii[1:], nptplus + 1)[1:,]
    radend = np.logspace(np.log10(radii[-1]), np.log10(rmax), 20)[1:]
    
    radplus = np.concatenate(
        [radstart, np.ravel(radmid, order="F"), radend]
    )
    
    rmeanplus = (radplus[1:] + radplus[:-1]) / 2.0
    nsym = int(np.floor(nptplus / 2))
    evalrad = (
        np.arange(nptplus + nsym - 1, nptplus + nsym + len(rmean) * nptplus, nptplus)
    )[: len(rmean)]
    return radplus, rmeanplus, evalrad


def WLmodel(WLdata, pmod, delta=200.0):
    """
    PyMC (Theano) model for predicting the mean tangential shear profile for a given density profile at a specified redshift.

    Parameters
    ----------
    WLdata : WLData
        Object containing all the necessary information about the cluster. This includes:
        radii_wl : The radial bins for the weak lensing data.
        rho_crit : The critical density at the cluster's redshift.
        msigmacrit : Mean inverse critical surface mass density for the cluster.
        fl : Second-order correction factor for weak lensing measurements.
    pmod : list
        List of parameters for the density profile model. For an NFW profile, this typically includes:
        cdelta : Concentration parameter.
        rdelta : Scale radius parameter.

    Returns
    -------
    gplus : numpy.ndarray
        Predicted mean tangential shear profile at the input radii.
    rm : numpy.ndarray
        Radii bins after applying interpolation or extrapolation through the function `get_radplus`.
    ev : numpy.ndarray
        Indices of the input data radii points within the new radii array, `rm`.
    """

    radplus, rm, ev = get_radplus(WLdata.radii_wl)

    rho_out = rho_nfw_cr(rm, pmod, delta) * WLdata.rho_crit

    sig = rho_to_sigma(radplus, rho_out)

    dsigma = dsigma_trap(sig, rm)

    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)

    return gplus, rm, ev


def WLmodel_np(WLdata, pmod, delta=200.0):
    """
    Numpy model for predicting the mean tangential shear profile for a given density profile at a specified redshift.

    Parameters
    ----------
    WLdata : WLData
        Object containing all the necessary information about the cluster. This includes:
        radii_wl : The radial bins for the weak lensing data.
        rho_crit : The critical density at the cluster's redshift.
        msigmacrit : Mean inverse critical surface mass density for the cluster.
        fl : Second-order correction factor for weak lensing measurements.
    pmod : list
        List of parameters for the density profile model. For an NFW profile, this typically includes:
        cdelta : Concentration parameter.
        rdelta : Scale radius parameter.

    Returns
    -------
    gplus : numpy.ndarray
        Predicted mean tangential shear profile at the input radii.
    rm : numpy.ndarray
        Radii bins after applying interpolation or extrapolation through the function `get_radplus`.
    ev : numpy.ndarray
        Indices of the input data radii points within the new radii array, `rm`.
    """
    radplus, rm, ev = get_radplus(WLdata.radii_wl)
    rho_out = rho_nfw_cr_np(radplus, pmod, delta=delta) * WLdata.rho_crit
    sig = rho_to_sigma_np(radplus, rho_out)
    dsigma = dsigma_trap_np(sig, radplus)
    gplus = get_shear(sig, dsigma, WLdata.msigmacrit, WLdata.fl)
    return gplus, rm, ev


class WLData:
    """
    A class to represent the weak lensing data for a galaxy cluster.

    Attributes
    ----------
    gplus : numpy.ndarray
        Mean tangential shear for the weak lensing data.
    err_gplus : numpy.ndarray
        Error on the mean tangential shear.
    rin_wl : numpy.ndarray
        Inner radial bin edges (in Mpc).
    rout_wl : numpy.ndarray
        Outer radial bin edges (in Mpc).
    radii_wl : numpy.ndarray
        Combined radial bin edges (inner + outer) for weak lensing data.
    rref_wl : numpy.ndarray
        Reference radius for each radial bin, defined as the average of rin_wl and rout_wl.
    rho_crit : float
        Critical density of the universe at the redshift of the cluster.
    msigmacrit : float
        Mean inverse critical surface mass density.
    fl : float
        Second-order correction factor for weak lensing.

    Parameters
    ----------
    redshift : float
        Redshift of the galaxy cluster.
    rin : numpy.ndarray, optional
        Inner radii (in arcminutes) for the weak lensing bins.
    rout : numpy.ndarray, optional
        Outer radii (in arcminutes) for the weak lensing bins.
    gplus : numpy.ndarray, optional
        Mean tangential shear measurements.
    err_gplus : numpy.ndarray, optional
        Errors on the mean tangential shear measurements.
    sigmacrit_inv : float, optional
        Mean inverse critical surface mass density.
    fl : float, optional
        Second-order correction factor (default is None, assuming first-order correction).
    cosmo : astropy.cosmology, optional
        Cosmological model to be used (default is Planck15).
    unit : str, optional
        Specifies whether the distances are in 'proper' or 'comoving' units (default is 'proper').

    Methods
    -------
    __init__ :
        Initializes the WLData object and computes radial bin edges and other derived attributes.
    """

    def __init__(
        self,
        redshift,
        rin=None,
        rout=None,
        gplus=None,
        err_gplus=None,
        sigmacrit_inv=None,
        fl=None,
        cosmo=None,
        unit="proper",
        delta=200.0,
    ):

        if rin is None or rout is None or gplus is None or err_gplus is None:

            print("Missing input, please provide rin, rout, gplus, and err_gplus")

            return

        if sigmacrit_inv is None:

            print("The mean value of sigma_crit is required")

            return

        if fl is None:

            print(
                "The second order correction factor is not given, we will do the calculation at first order"
            )

        self.gplus = gplus

        self.err_gplus = err_gplus

        if cosmo is None:

            from astropy.cosmology import Planck15 as cosmo

        if unit == "proper":
            amin2kpc = cosmo.kpc_proper_per_arcmin(redshift).value
        if unit == "comoving":
            amin2kpc = cosmo.kpc_comoving_per_arcmin(redshift).value

        self.rin_wl = rin * amin2kpc / 1e3  # Mpc

        self.rout_wl = rout * amin2kpc / 1e3

        self.radii_wl = np.append(self.rin_wl[0], self.rout_wl)

        self.rin_wl_am = rin

        self.rout_wl_am = rout

        self.rref_wl = (self.rin_wl + self.rout_wl) / 2.0

        self.rho_crit = (cosmo.critical_density(redshift).to(u.M_sun * u.Mpc**-3)).value

        self.msigmacrit = sigmacrit_inv

        self.fl = fl

        self.delta = delta
