"""
   hods_utils.py --- General utilities for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: ---

"""

import numpy as np
from scipy import integrate
import hankel
import camb

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


def get_camb_pk(redshift=0,
                OmegaM0=0.27, OmegaL0=0.73, OmegaB0=0.04,
                H0=70.0, Pinit_As = 2.2e-9, Pinit_n = 0.96,
                nonlinear=False, halofit_model=None,
                kmin=1e-4, kmax=20.0, k_per_logint=10):
    """
    Function that obtains a sampled matter Power Spectrum for a set of
    parameters using the CAMB library. We print also the value of \sigma_8
    obtained for these parameters (comparable to the 'standard' one if
    redshift=0 and nonlinear=False).

    Parameters:
    - redshift: redshift for which we calculate the P(k). It should be a single
                value.
    - Cosmological parameters to pass to CAMB (OmegaM0, OmegaL0, OmegaB0,
        H0, Pinit_As, Pinit_n): will convert to 'physical densities' when
        needed. We assume Omega_nu=0.
    - nonlinear: whether to compute the non-linear P(k) (using HaloFit)
    - halofit_model: If nonlinear, which HaloFit version to use. If it is None,
        use CAMB's default model (currently, `mead`). See documentation for
        `camb.nonlinear.Halofit.set_params` for valid options.
    - kmin, kmax, k_per_logint: parameters defining the output array in k, following
        CAMB's terminology

    Output:
    - A PowerSpectrum object
    """

    # Get default set of CAMB Parameters
    params = camb.CAMBparams()

    # Set cosmology parameters
    h = H0/100
    ombh2 = OmegaB0*h*h
    omch2 = (OmegaM0 - OmegaB0)*h*h
    omk = 1 - OmegaM0 - OmegaL0
    if np.abs(omk) < 1e-5:
        omk = 0
    if omk != 0:
        print(f"You are using a non-flat cosmology (OmegaK = {omk})."
              "Are you sure this is what you really want?")
#        raise UserWarning("You are using a non-flat cosmology. \
#            Are you sure that is what you really want?")

    params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, mnu=0)
    params.InitPower.set_params(As=Pinit_As, ns=Pinit_n)

    # Set non-linearity parameters
    if nonlinear:
        nl_label = "Non-linear"
        params.NonLinear = camb.model.NonLinear_both
        if halofit_model is not None:
            params.NonLinearModel.set_params(halofit_version=halofit_model)
    else:
        nl_label = "Linear"
        params.NonLinear = camb.model.NonLinear_none

    # Set parameters for P(k)
    params.set_matter_power(redshifts=(redshift,), kmax=kmax,
                            k_per_logint=k_per_logint)

    # Do the calculation
    # results = camb.get_results(params)
    results = camb.get_transfer_functions(params)
    npoints = (np.log(kmax) - np.log(kmin))*k_per_logint
    npoints = np.int(npoints)
    kh, _, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=npoints)
    sigma8 =  results.get_sigma8()

    print(f"{nl_label} power spectrum calculated at z={redshift} for the given parameters.")
    print(f"For this P(k) we obtain \sigma_8={sigma8[0]:.5}")

    # Get the output as a PowerSpectrum object
    return PowerSpectrum(kvals=kh, pkvals=pk[0])
