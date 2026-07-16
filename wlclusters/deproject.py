import numpy as np

def deproj_vol(radin, radot):
    """
    Calculate the deprojected volume matrix for spherical shells.

    This method performs a deprojection of volumes for the given inner (`radin`)
    and outer (`radot`) radii. It adjusts for discrepancies between adjacent radii
    and ensures consistent volume calculations for each shell.
    
    Args:
        radin (numpy.ndarray): An array of inner radii for the shells.
        radot (numpy.ndarray): An array of outer radii for the shells.

    Returns:
        numpy.ndarray: A matrix where each element [i, j] represents the deprojected
        volume of the j-th shell inside the i-th ring.

    Notes:
        - The method first checks for discrepancies between adjacent radii and fixes
            them if necessary.
        - A warning is printed if the discrepancies exceed a 0.1% threshold.
        - The volume matrix is then computed using the provided inner and outer radii.

    Raises:
        SystemExit: If any computed volume element is negative, the program will exit.
    """
    ri = np.copy(radin)
    ro = np.copy(radot)

    diftot = 0
    for i in range(1, len(ri)):
        dif = abs(ri[i] - ro[i - 1]) / ro[i - 1] * 100.0
        diftot += dif
        ro[i - 1] = ri[i]

    if abs(diftot) > 0.1:
        print(
            " DEPROJ_VOL: WARNING - abs(ri(i)-ro(i-1)) differs by",
            diftot,
            " percent",
        )
        print(" DEPROJ_VOL: Fixing up radii ... ")
        for i in range(1, len(ri) - 1):
            dif = abs(ri[i] - ro[i - 1]) / ro[i - 1] * 100.0
            diftot += dif

    nbin = len(ro)
    volconst = 4.0 / 3.0 * np.pi
    volmat = np.zeros((nbin, nbin))
    
    diag_idx = np.arange(nbin)
    volmat[diag_idx, diag_idx] = (
        volconst * ro**3 * (1.0 - (ri / ro) ** 2) ** 1.5
    )

    ishell, iring = np.tril_indices(nbin, k=-1)
    
    ri_iring = ri[iring]
    ro_iring = ro[iring]
    ri_ishell = ri[ishell]
    ro_ishell = ro[ishell]
    
    f1 = (1.0 - (ri_iring / ro_ishell) ** 2) ** 1.5 - \
         (1.0 - (ro_iring / ro_ishell) ** 2) ** 1.5

    f2 = (1.0 - (ri_iring / ri_ishell) ** 2) ** 1.5 - \
         (1.0 - (ro_iring / ri_ishell) ** 2) ** 1.5
         
    vals = volconst * (f1 * ro_ishell**3 - f2 * ri_ishell**3)
    if np.any(vals < 0.0):
        raise ValueError("The computed volume elements contain negative values. Check the input radii.")
    volmat[iring, ishell] = vals

    return volmat
