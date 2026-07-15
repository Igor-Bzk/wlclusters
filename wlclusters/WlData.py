import numpy as np
import astropy.units as u

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

    def get_radplus(self, rmin=1e-3, rmax=1e2, nptplus=19):
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
            rmean = (self.radii_wl[1:] + self.radii_wl[:-1]) / 2.0

            radstart = np.logspace(np.log10(rmin), np.log10(self.radii_wl[0]), nptplus)
            radmid = np.linspace(self.radii_wl[:-1], self.radii_wl[1:], nptplus + 1)[1:,]
            radend = np.logspace(np.log10(self.radii_wl[-1]), np.log10(rmax), 20)[1:]
            
            radplus = np.concatenate(
                [radstart, np.ravel(radmid, order="F"), radend]
            )
            
            rmeanplus = (radplus[1:] + radplus[:-1]) / 2.0
            nsym = int(np.floor(nptplus / 2))
            evalrad = (
                np.arange(nptplus + nsym - 1, nptplus + nsym + len(rmean) * nptplus, nptplus)
            )[: len(rmean)]
            
            rmean2 = (rmeanplus[1] + rmeanplus[0]) / 2
            dr = rmeanplus[1:] - rmeanplus[:-1]
            return radplus, rmeanplus, evalrad, rmean2, dr