import numpy as np
from .deproject import deproj_vol
import pymc as pm
from .WlData import WLData

from abc import ABC, abstractmethod

class WlModel(ABC):
    def __init__(self, WLdata : WLData, parnames = ["cdelt", "mdelt"], delta=200.0):
        self.rho_crit = WLdata.rho_crit
        self.msigmacrit = WLdata.msigmacrit
        self.fl = WLdata.fl
        self.delta = delta
        self.parnames = parnames
        
        self.pi_delta_rhoc = (4 / 3) * np.pi * delta * WLdata.rho_crit
        
        radplus, self.rmean, self.evalrad, self.rmean2, self.dr = WLdata.get_radplus()
        
        self.setup_projection(radplus)

    # ==================== SETUP SECTION ====================
    
    def setup_projection(self, radplus):
        self.proj_vol = deproj_vol(radplus[:-1], radplus[1:])
        radbins_proj = (radplus * 1e6) ** 2
        self.area_proj = np.pi * (radbins_proj[1:] - radbins_proj[:-1])

    # ==================== MODELING SECTION ====================

    def run(self, pmod=None):
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
        if pmod is not None:
            cdelt, rdelt = pmod
        else:
            cdelt, rdelt = self.setup_parameters()
        
        rho_out = self.rho_nfw_cr(cdelt, rdelt) * self.rho_crit

        sig = self.rho_to_sigma(rho_out)

        dsigma = self.dsigma_trap(sig)

        gplus = self.get_shear(sig, dsigma)

        return gplus, self.rmean, self.evalrad

    def setup_parameters(self):
        """
        Sets up the parameters for the NFW profile model based on the chosen parameterization by converting the user choice into cdelt and rdelt.

        Args:
            parnames (list): List of parameter names to be used in the model (e.g. ['cdelt', 'rdelt'], ['cdelt', 'mdelt'], etc.).
            cosmo (astropy.cosmology.Cosmology): The cosmology object to be used for calculations.
            clust_z (float):The redshift of the current cluster.

        Returns:
            list: List of parameters for the model.
        """
        match self.parnames[0]:
            case "cdelt":
                cdelt = pm.Uniform(name="cdelt", lower=1.0, upper=10.0)
            case "log10cdelt":
                cdelt = pm.Uniform(name="log10cdelt", lower=0.0, upper=1.0)
            case _:
                raise ValueError("Invalid parnames specified.")
        
        match self.parnames[1]:
            case "rdelt":
                rdelt = pm.Uniform(name="rdelt", lower=200.0, upper=4000.0)
            case "mdelt":
                mdelt = pm.Uniform(name="mdelt", lower=1e12, upper=1e16)
                rdelt = pm.Deterministic("rdelt", self.mdelt_to_rdelt(mdelt))
            case "log10mdelt":
                log10mdelt = pm.Uniform(name="log10mdelt", lower=12.0, upper=16.0)
                mdelt = pm.Deterministic("mdelt", 10**log10mdelt)
                rdelt = pm.Deterministic("rdelt", self.mdelt_to_rdelt(mdelt))
            case _:
                raise ValueError("Invalid parnames specified.")
        return [cdelt, rdelt]

    def rdelt_to_mdelt(self, r):
        """
        Convert radius `r_delta` to mass `m_delta` for a given redshift and cosmology.

        Args:
            r (float): The radius `r_delta` in kpc.
            z (float): The redshift of the cluster.
            cosmo (astropy.cosmology.Cosmology): Cosmology object used for calculations (e.g., Planck15).
            delta (float, optional): Overdensity factor (default is 200, corresponding to `r200`).

        Returns:
            float: The mass `m_delta` corresponding to the given radius `r_delta` at redshift `z` and overdensity `delta`.
        """
        return self.pi_delta_rhoc * r**3

    def mdelt_to_rdelt(self, m):
        """
        Convert mass `m_delta` to radius `r_delta` for a given redshift and cosmology.

        Args:
            m (float): The mass `m_delta` in solar masses.
            z (float): The redshift of the cluster.
            cosmo (astropy.cosmology.Cosmology): Cosmology object used for calculations (e.g., Planck15).
            delta (float, optional): Overdensity factor (default is 200, corresponding to `m200`).

        Returns:
            float: The radius `r_delta` corresponding to the given mass `m_delta` at redshift `z` and overdensity `delta`.
        """
        return (m / self.pi_delta_rhoc) ** (1 / 3)

    def rho_nfw_cr(self, cdelt, rdelt):
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
        r = self.rmean * 1000.0  # Convert radii to kpc

        # Calculate delta_crit using PyMC math functions
        delta_crit = (
            (self.delta / 3)
            * (cdelt**3)
            * (self.log(1.0 + cdelt) - cdelt / (1 + cdelt)) ** (-1)
        )
        # Return NFW density profile
        cdelt_r_delt = cdelt * r / rdelt
        return delta_crit / (cdelt_r_delt * ((1.0 + cdelt_r_delt) ** 2))

    def rho_to_sigma(self, rho):
        """
        Projects a 3D density profile to compute the surface mass density using PyMC (Theano).

        Args:
            radii_bins (array): Binned radial distances.
            rho (array): 3D density profile values.

        Returns:
            array: Projected surface mass density in units of M_sun * Mpc**-2.
        """
        sigma = self.dot(self.proj_vol, rho) / self.area_proj
        return sigma * 1e12

    def dsigma_trap(self, sigma):
        """
        Computes Delta Sigma, the differential surface mass density, using numerical trapezoidal integration.

        Args:
            sigma (array): Projected surface mass density values.
            radii (array): Radial distances.

        Returns:
            array: Differential surface mass density (Delta Sigma).
        """

        arg0 = sigma[0] * (self.rmean2 ** 2) / 2
        sigma_rmean = sigma * self.rmean
        arg1 = self.dr * (sigma_rmean[1:] + sigma_rmean[:-1]) / 2

        arg = self.concatenate([[arg0], arg1])
        a = self.cumsum(arg)
        sigmabar = (2 / (self.rmean**2)) * a
        dsigma = sigmabar - sigma
        return dsigma

    def get_shear(self, sigma, dsigma):
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

        shear = (dsigma * self.msigmacrit) / (1 - self.fl * sigma * self.msigmacrit)

        return shear
    
    # ==================== ABSTRACT METHODS ====================
    # Used for easily switching between PyMC and Numpy
    
    @abstractmethod
    def cumsum(self, array):
        pass
    
    @abstractmethod
    def concatenate(self, arrays):
        pass
    
    @abstractmethod
    def dot(self, a, b):
        pass
    
    @abstractmethod
    def log(self, x):
        pass