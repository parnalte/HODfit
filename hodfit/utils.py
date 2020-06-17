"""
   hods_utils.py --- General utilities for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: ---

"""

import numpy as np
from scipy import integrate
import hankel

# GLOBAL CONSTANTS

# Critical density in units of
# h^2 M_sol Mpc^-3 (from Peacock book)
# Independent of cosmology, just definition of rho_crit
# and using H0 and G in adequate units
# This is adequate for the units used here
RHO_CRIT_UNITS = 2.7752E+11


def xi2d_iso(rp, pi, xirfunc):

    rval = np.sqrt((rp*rp) + (pi*pi))

    return xirfunc(rval)


def xir2wp_xifunc_pi(rpvals, xirfunc, logpimin=-4, logpimax=3, npi=500):
    """
    Get the value of the projected correlation function wp at a projected
    separation rp from a xi(r) defined by function 'xirfunc'.

    We assume isotropy and integrate along the pi direction (using a
    log-space pi array defined by the corresponding parameters).
    This is better than using the direct r-->rp conversion (e.g. eq. 2.3
    in thesis), as we avoid having a divergence at r=rp.

    Will use Simpson's integration along the pi direction, doing the
    integration from -pi_max to pi_max (i.e. taking also negative pi)
    in order to better sample the region around pi=0.

    rp can be an array (although for the moment we deal with it in
    the slow non-pythonic way)
    """

    # First, checks on rp and convert to 1D array
    rpvals = np.atleast_1d(rpvals)
    assert rpvals.ndim == 1
    Nrp = len(rpvals)

    wp_out = np.empty(Nrp, float)

    piarr = np.logspace(logpimin, logpimax, npi)

    # Start loop over rpvalues
    for i, rp in enumerate(rpvals):

        # Get values of xi
        xi2d_arr = xi2d_iso(rp=rp, pi=piarr, xirfunc=xirfunc)

        # Construct arrays for both negative and positive pi
        piarr_full = np.concatenate((-piarr[::-1], piarr))
        xi2darr_full = np.concatenate((xi2d_arr[::-1], xi2d_arr))

        # Do the actual integration
        wp_out[i] = integrate.simps(y=xi2darr_full, x=piarr_full, even='first')

    return wp_out


def xir2wp_pi(rpvals, rvals, xivals, logpimin=-4, logpimax=3, npi=500):
    """
    Wrapper over the function xir2wp_xifunc_pi for the case in which xi(r)
    is defined by a set of tabulated values. We just create a linear
    interpolator of it and pass it to the function.
    """

    # Now, checks on tabulated values of xi(r)
    if not (sorted(rvals) == rvals).all():
        raise UserWarning("In function xir2wp_pi: "
                          "the values of r you gave me were not sorted!")

        idx_sort = rvals.argsort()
        rvals = rvals[idx_sort]
        xivals = xivals[idx_sort]

    xifunc = lambda x: np.interp(x, rvals, xivals)

    return xir2wp_xifunc_pi(rpvals=rpvals, xirfunc=xifunc, logpimin=logpimin,
                            logpimax=logpimax, npi=npi)


class PowerSpectrum(object):
    """Simple class to contain a power spectrum sampled at a set of
       given values of the wavenumber k
    """

    def __init__(self, kvals, pkvals):
        """
        """

        if(kvals.ndim != 1 or pkvals.ndim != 1):
            raise TypeError("The k and pk values passed to the PowerSpectrum "
                            "class should be 1-dimensional arrays!")

        self.N = len(kvals)

        if(self.N != len(pkvals)):
            raise ValueError("The k and pk arrays passed to the PowerSpectrum "
                             "class do not have same length!")

        if((kvals < 0).any()):
            raise ValueError("The k values passed to the PowerSpectrum class "
                             "must be positive!")

        if((pkvals < 0).any()):
            raise ValueError("The Pk values passed to the PowerSpectrum class "
                             "must be positive!")

        sortind = kvals.argsort()

        if((sortind != list(range(self.N))).any()):
            self.k = kvals[sortind]
            self.pk = pkvals[sortind]
            raise UserWarning("k-values passed to PowerSpectrum class were "
                              "not correctly ordered!")
        else:
            self.k = kvals
            self.pk = pkvals

    def pkinterp(self, kval):
        """
        Function that interpolates linearly the power spectrum using the
        given values.
        """

        return np.interp(x=kval, xp=self.k, fp=self.pk)

    def xir(self, rvals, hankelN=6000, hankelh=0.0005, ft_hankel=None):
        """
        Function that performs a Hankel transform to obtain the 2-point
        correlation function corresponding to the power spectrum.
        """

        if ft_hankel is None:
            ft_hankel = hankel.SymmetricFourierTransform(ndim=3,
                                                         N=hankelN,
                                                         h=hankelh)
        xivals = ft_hankel.transform(f=self.pkinterp, k=rvals, ret_err=False,
                                     ret_cumsum=False, inverse=True)

        return xivals
