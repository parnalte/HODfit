"""
hods_densprofile.py --- Profile-related function/classes for the simple
                        HOD model

Author: P. Arnalte-Mur (ICC-Durham)
Creation date: 19/02/2014
Last modified: --

This module will contain the classes and functions related to the halo
density profile. We base everything on the NFW profile, as defined in
Coupon et al., 2012 (C2012) and Cooray and Sheth, 2002 (CS02).

"""


import numpy as np
import astropy.cosmology as ac
import scipy.special as spc
from scipy import integrate
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import hankel

import halomodel
from utils import RHO_CRIT_UNITS


def Delta_vir(redshift=0, cosmo=ac.WMAP7):
    """ Computes the Delta_vir(z) function, as defined by eq.
        (A12) in C2012.

        This function works OK when input 'redshift' is an array.
    """
    fact1 = 18*np.pi*np.pi
    fact2 = 0.399
    expon = 0.941
    Omz = cosmo.Om(redshift)

    result = fact1*(1. + (fact2*pow((1./Omz) - 1., expon)))

    return result


def rvir_from_mass(mass=1e10, redshift=0, cosmo=ac.WMAP7):
    """Obtain the virial radius from the halo mass. From the
       inversion of equation (A.11) in C2012.

       This function works OK when EITHER input 'mass' or 'redshift' are
       1D arrays, but NOT when BOTH of them are arrays.
    """

    rho_mean_0 = RHO_CRIT_UNITS*cosmo.Om0
    Dvir = Delta_vir(redshift=redshift, cosmo=cosmo)

    result = pow(3.*mass/(4.*np.pi*rho_mean_0*Dvir), 1./3.)

    return result


def massvir_from_radius(radius=8.0, redshift=0, cosmo=ac.WMAP7):
    """
    Obtain the virial mass corresponding to a given radius.
    From equation (A.11) in C2012.

    This function works OK when EITHER input 'radius' or 'redshift' are
    1D arrays, but NOT when BOTH of them are arrays.
    """

    rho_mean_0 = RHO_CRIT_UNITS*cosmo.Om0
    Dvir = Delta_vir(redshift=redshift, cosmo=cosmo)

    result = 4.*np.pi*rho_mean_0*Dvir*(radius**3.)/3.

    return result


def mstar_interp(cosmo=ac.WMAP7, powesp_lin_0=None, logM_min=10.0,
                 logM_max=16.0, logM_step=0.05):
    """Obtain M_* from the condition sigma(M_*) = delta_c(z=0).
       Will get it from a linear interpolation to the function sigma(M).
    """

    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    # sigma_mass() already works well with input mass arrays
    sigma_array = halomodel.sigma_mass(mass=mass_array, cosmo=cosmo,
                                       powesp_lin_0=powesp_lin_0)

    delta_c0 = halomodel.delta_c_z(redshift=0, cosmo=cosmo)

    # We want to compute the function M(sigma) at the point sigma=delta_c0
    # First, need to sort the 'x' array, in this case sigma
    idx_sort = np.argsort(sigma_array)
    mass_star = np.interp(x=delta_c0, xp=sigma_array[idx_sort],
                          fp=mass_array[idx_sort])

    return mass_star


def concentration(mass=1e10, redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None,
                  c_zero=11.0, beta=0.13, logM_min=10.0, logM_max=16.0,
                  logM_step=0.05):
    """Concentration for a halo of a given mass, following eq. (A10)
       in C2012. Need mass array to obtain Mstar via interpolation

       This function works OK when EITHER input 'mass' or 'redshift' are
       1D arrays, but NOT when BOTH of them are arrays.
    """

    mass_star = mstar_interp(cosmo=cosmo, powesp_lin_0=powesp_lin_0,
                             logM_min=logM_min, logM_max=logM_max,
                             logM_step=logM_step)

    conc = (c_zero/(1. + redshift))*pow(mass/mass_star, -beta)

    return conc


def rhos_from_charact(mass=1e10, rvir=1.0, conc=10.0):
    """
    Obtain the normalization parameter, rho_s, given the characteristics of
    the profile (r_vir, concentration), and the total mass enclosed (mass).
    This follows from eq. (A9) in C2012.
    """

    term1 = 4.*np.pi*pow(rvir, 3)/pow(conc, 3)
    term2 = np.log(1. + conc) - (conc/(1.+conc))

    rho_s = mass/(term1*term2)

    return rho_s


def profile_NFW_config_parameters(rvals, rho_s, conc, rvir):
    """
    Function that returns the standard NFW profile in configuration space
    (as function of scale 'rvals'), given the basic parameters
    rho_s (normalization), conc (concentration) and rvir (virial radius).

    We truncate the resulting profile at r=rvir.

    Assume that rho_s and conc are arrays of length Nm, corresponding to Nm
    different haloes of several masses.

    Returns an array of shape (Nr, Nm), where Nr is the length of rvals.
    """

    rvals = np.atleast_1d(rvals)
    assert rvals.ndim == 1

    rho_s = np.atleast_1d(rho_s)
    conc = np.atleast_1d(conc)
    assert rho_s.ndim == 1
    assert conc.ndim == 1
    assert len(rho_s) == len(conc)

    r_s = rvir/conc

    fact1 = np.outer(rvals, 1./r_s)
    fact2 = pow(1. + fact1, 2.0)
    rho_h = rho_s/(fact1*fact2)

    rvir_grid, rvals_grid = np.meshgrid(rvir, rvals)
    rho_h[rvals_grid > rvir_grid] = 0
    return rho_h


def profile_ModNFW_config_parameters(rvals, rho_s, conc, rvir, gamma=1):
    """
    Function that returns the *modified* NFW profile in configuration space
    (as function of scale 'rvals'), given the basic parameters
    rho_s (normalization), conc (concentration), rvir (virial radius),
    and gamma (inner slope).

    We truncate the resulting profile at r=rvir.

    Assume that rho_s, conc and rvir are arrays of length Nm, corresponding to
    Nm different haloes of several masses.

    Returns an array of shape (Nr, Nm), where Nr is the length of rvals.
    """

    rvals = np.atleast_1d(rvals)
    assert rvals.ndim == 1
    # print rvals.shape

    rho_s = np.atleast_1d(rho_s)
    conc = np.atleast_1d(conc)
    rvir = np.atleast_1d(rvir)
    assert rho_s.ndim == 1
    assert conc.ndim == 1
    assert rvir.ndim == 1
    Nm = len(rho_s)
    assert Nm == len(conc) == len(rvir)

    r_s = rvir/conc

    r_ratios = np.divide.outer(rvals.T, r_s)
    fact1 = (r_ratios)**gamma
    fact2 = (1. + r_ratios)**(3. - gamma)
    rho_h = rho_s/(fact1*fact2)

#    rvir_grid, rvals_grid = np.meshgrid(rvir, rvals)
#    rho_h[rvals_grid > rvir_grid] = 0
    idx_zero = np.greater.outer(rvals.T, rvir)
    # print rho_h.shape
    # print idx_zero.shape
    rho_h[idx_zero] = 0

    return rho_h


def profile_ModNFW_config_scalar(rvals, rho_s, conc, rvir, gamma_exp=1):
    """
    (TEMP) Same as above, but now:
        - Assumes rho_s, conc, rvir will be scalars (single value of each)
        - rvals can be an array of any dimension (in particular ndim=2)
    This is to allow this to work with the new SymmetricFourierTransform in
    hankel library
    """

    idx_calc = (rvals < rvir)
    rho_h = np.zeros_like(rvals)

    r_ratios = rvals[idx_calc]*conc/rvir

    fact1 = r_ratios**gamma_exp
    fact2 = (1. + r_ratios)**(3. - gamma_exp)
    rho_h[idx_calc] = rho_s/(fact1*fact2)

    # rho_h[rvals > rvir] = 0
    return rho_h


def integrand_unnorm(r, conc, rvir, gamma):
    """
    Integrand function needed for the integral performed in
    rhos_from_charact_modNFW , to obtain the mass corresponding to an
    unnormalized ModNFW profile.

    Returns an array of shape (Nm, Nr).
    """
    conc = np.atleast_1d(conc)
    rvir = np.atleast_1d(rvir)
    assert conc.ndim == 1
    assert rvir.ndim == 1
    Nm = len(conc)
    assert Nm == len(rvir)

    rho_u = profile_ModNFW_config_parameters(rvals=r, rho_s=np.ones(Nm),
                                             conc=conc, rvir=rvir, gamma=gamma)
    return 4*np.pi*r*r*rho_u.T


def rhos_from_charact_modNFW(mass=1e10, rvir=1.0, conc=10.0, gamma=1.0):
    """
    Obtain the normalization parameter, rho_s, given the characteristics of
    the profile (r_vir, concentration, gamma), and the total mass enclosed
    (mass).

    We do this for the generalized NFW profile, so we do the integral of the
    profile numerically to get the normalization.
    """

    mass = np.atleast_1d(mass)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rvir) == len(conc)

    # Define r-array to do the numerical (Simpson) integration
    # TODO: add options to define this array, and to test the results
    # For now, using rmin=1e-5, and Nr=1000, I get that the maximum relative
    # error in rho_s for gamma=1 (where I can compare to analytic formula)
    # is 5e-3
    max_rvir = rvir.max()
    r_vals_array = np.logspace(-5, np.log10(max_rvir), 1000)

    # Do numerical integral and get the normalization
    integ_unnorm_vals = integrand_unnorm(r=r_vals_array, conc=conc, rvir=rvir,
                                         gamma=gamma)
    integral_unnorm = integrate.simps(y=integ_unnorm_vals,
                                      x=r_vals_array,
                                      even='first')

    # Add step to smooth the results, when Nm is large enough.
    # With this, typically the error in rho_s (for gamma=1) goes always below
    # 5e-5 for logM=[9.5,16.5], or below 3e-4 for logM=[8,17]
    # Alternative (quad integration) would go down to ~5e-7, but much slow
    # (need explicit loop over Nm)
    # Comparing to alternative:
    # * gamma=0.5: <5e-5 for logM=[8,16], <2e-4 for logM=[8,17]
    # * gamma=2.0: <5e-4 for logM=[11,17], <~0.01 (1%) for logM=[8,17] (!)
    if Nm >= 10:
        # Check mass array is sorted
        assert (mass == np.sort(mass)).all()

        # Define interpolator spline (in log-space)
        us = UnivariateSpline(x=np.log10(mass), y=np.log10(integral_unnorm),
                              k=5)

        # Get smoothed values
        integral_unnorm_smooth = 10**us(np.log10(mass))

        return mass/integral_unnorm_smooth

    # If Nm is small, just return raw value, without smoothing
    else:
        return mass/integral_unnorm


def alt_rhos_modNFW(mass=1e10, rvir=1.0, conc=10.0, gamma=1.0):
    """
    Alternative to rhos_from_charact_modNFW, using a much precise integration
    (~5e-7 over all range, ~1e-10 for logM>11), but extremely slow
    (due to the need for an explicit loop over mass).

    I won't use this function, but leave it here for comparison purposes.
    """
    mass = np.atleast_1d(mass)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rvir) == len(conc)

    integral_unnorm_m = np.empty(Nm)

    for i in range(Nm):
        integral_unnorm_m[i] = \
            integrate.quad(integrand_unnorm, a=0, b=rvir[i],
                           args=(conc[i], rvir[i], gamma))[0]

    return mass/integral_unnorm_m


def profile_NFW_fourier_parameters(kvals, mass, rho_s, rvir, conc):
    """
    Function that returns the standard NFW profile in Fourier space
    (as function of wavenumber 'kvals'), given the basic parameters
    of the haloes.

    We take this from eq. (81) in CS02, so it already takes into account
    the truncation of the halo.

    Assume that mass, rho_s, rvir and conc are arrays of length Nm,
    corresponding to Nm different haloes of several masses.

    Returns an array of shape (Nk, Nm), where Nk is the length of kvals.
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1

    mass = np.atleast_1d(mass)
    rho_s = np.atleast_1d(rho_s)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    assert len(mass) == len(rho_s) == len(rvir) == len(conc)

    r_s = rvir/conc

    # Need to compute the sine and cosine integrals
    si_ckr, ci_ckr = spc.sici(np.outer(kvals, (1. + conc) * r_s))
    si_kr, ci_kr = spc.sici(np.outer(kvals, r_s))

    fact1 = np.sin(np.outer(kvals, r_s)) * (si_ckr - si_kr)
    fact2 = np.sin(np.outer(kvals, conc * r_s)) / \
        (np.outer(kvals, (1. + conc) * r_s))
    fact3 = np.cos(np.outer(kvals, r_s)) * (ci_ckr - ci_kr)

    uprof = 4.*np.pi*(rho_s / mass) * pow(r_s, 3.) * (fact1 - fact2 + fact3)

    return uprof


def profile_ModNFW_fourier_hankel_interp(kvals, mass, rho_s, rvir, conc,
                                         gamma=1.0, hankelN=6000, hankelh=1e-5,
                                         ft_hankel=None, Nk_interp=None,
                                         Nm_interp=None):
    """
    Function to calculate the Fourier-space profile of the ModNFW profile.

    In this case, we allow for the use of interpolation in both kvals
    and the masses, so that we do the costly calculation (given by the
    function profile_ModNFW_fourier_hankel) only for a coarser grid in
    k,mass, and then interpolate to obtain the needed values.
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1
    Nkin = len(kvals)

    mass = np.atleast_1d(mass)
    assert mass.ndim == 1
    Nmin = len(mass)

    logkvals_in = np.log10(kvals)
    logmass_in = np.log10(mass)

    if (Nk_interp is not None) and (Nk_interp < Nkin):
        Nkout = Nk_interp
    else:
        Nkout = Nkin

    logkvals_interp = np.linspace(np.log10(kvals.min()), np.log10(kvals.max()),
                                  Nkout)
    kvals_interp = 10**logkvals_interp

    if (Nm_interp is not None) and (Nm_interp < Nmin):
        Nmout = Nm_interp
    else:
        Nmout = Nmin

    assert Nkout > 0
    assert Nmout > 0

    # General case
    if Nmout > 1:
        logmassvals_interp = np.linspace(np.log10(mass.min()),
                                         np.log10(mass.max()), Nmout)
        mass_interp = 10**logmassvals_interp

        # Get the corresponding values of rho_s, rvir, conc by linear
        # interpolation
        # TODO: improve this?
        rho_s_spline = UnivariateSpline(x=logmass_in, y=rho_s, k=1, s=0)
        rho_s_interp = rho_s_spline(logmassvals_interp)

        rvir_spline = UnivariateSpline(x=logmass_in, y=rvir, k=1, s=0)
        rvir_interp = rvir_spline(logmassvals_interp)

        conc_spline = UnivariateSpline(x=logmass_in, y=conc, k=1, s=0)
        conc_interp = conc_spline(logmassvals_interp)

        # Do the actual calculation in the coarser grid
        uprof_interp = profile_ModNFW_fourier_hankel(kvals_interp, mass_interp,
                                                     rho_s_interp, rvir_interp,
                                                     conc_interp, gamma,
                                                     hankelN, hankelh,
                                                     ft_hankel)
        if Nkout > 1:

            # And now interpolate to obtain the values at the desired points
            # for now, simply linear interpolation.
            # TODO: revise this?)
            uprof_spline_2d = RectBivariateSpline(x=logkvals_interp,
                                                  y=logmassvals_interp,
                                                  z=uprof_interp,
                                                  kx=1, ky=1, s=0)
            uprof_output = uprof_spline_2d(x=logkvals_in, y=logmass_in,
                                           grid=True)

        # Nk == 1
        else:
            uprof_spline_1d_m = UnivariateSpline(x=logmassvals_interp,
                                                 y=uprof_interp[0], k=1, s=0)
            uprof_output_mass = uprof_spline_1d_m(x=logmass_in)
            uprof_output = np.expand_dims(uprof_output_mass, 0)

    # Nm == 1
    else:

        # In this case, calculate directly at input (single) values for mass
        # and related quantities
        uprof_interp = profile_ModNFW_fourier_hankel(kvals_interp, mass, rho_s,
                                                     rvir, conc, gamma,
                                                     hankelN, hankelh,
                                                     ft_hankel)
        # Need to interpolate over k
        if Nkout > 1:
            uprof_spline_1d_k = UnivariateSpline(x=logkvals_interp,
                                                 y=uprof_interp[:, 0],
                                                 k=1, s=0)
            uprof_output_k = uprof_spline_1d_k(x=logkvals_in)
            uprof_output = np.expand_dims(uprof_output_k, 1)

        # Nk == 1
        else:
            # In this case, single value of (k, mass), no need to interpolat
            # anything
            uprof_output = uprof_interp

    return uprof_output


def profile_ModNFW_fourier_hankel(kvals, mass, rho_s, rvir, conc,
                                  gamma=1.0, hankelN=6000, hankelh=1e-5,
                                  ft_hankel=None):
    """
    Try to do the same as the function profile_ModNFW_fourier_parameters,
    but using the Hankel transforms in the 'hankel' library.
    We use the fact that the transform we need to do is the same as to
    convert from xi(r) to P(k), assuming that the rho(r) function provides
    already a truncated profile.

    For now, using N=6000, h=1e-5, and our cut at k=1/(10 rvir),
    I get a difference typically below 1.5% comparing to the u(k) calculated
    analytically for gamma=1.

    Could in principle use the multidimensional version of hankel to speed
    up the transformation (avoiding explicit loop over Nm).
    However, for useful values of Nm this rockets up the memory usage, so this
    approach is not viable in practice.
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1
    Nk = len(kvals)

    mass = np.atleast_1d(mass)
    rho_s = np.atleast_1d(rho_s)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rho_s) == len(rvir) == len(conc)

    if ft_hankel is None:
        ft_hankel = hankel.SymmetricFourierTransform(ndim=3, N=hankelN,
                                                     h=hankelh)
    uprof_out = np.ones((Nk, Nm), float)
    for i in range(Nm):
        idx_calc = (kvals > 1./(10*rvir[i]))
        norm_prof_func = lambda x: \
            profile_ModNFW_config_scalar(rvals=x, rho_s=rho_s[i],
                                         conc=conc[i], rvir=rvir[i],
                                         gamma_exp=gamma)/mass[i]
        uprof_out[idx_calc, i] = ft_hankel.transform(f=norm_prof_func,
                                                     k=kvals[idx_calc],
                                                     ret_err=False,
                                                     ret_cumsum=False)
    # I add a condition, so that I make u(k)=1 for
    # all k < 1/(10 rvir).
    # This is a bit arbitrary, what works to avoid the oscillations
    # that appear at small k
    # rvir_grid, kvals_grid = np.meshgrid(rvir, kvals)
    # uprof_out[kvals_grid < 1./(10*rvir_grid)] = 1.0

    return uprof_out


def create_profile_grid_fourier(log_kvals_rvir_dict, log_conc_dict,
                                gamma_dict, output_file="default.npz",
                                hankelN=12000, hankelh=1e-6, verbose=True):
    """
    Function to create and save to memory a grid of Fourier-space density
    profiles, so that we can later use them to speed-up calculations
    using interpolations.
    The resulting grid will depend on three coordinates:
        - log10(k * rvir)
        - log10(concentration)
        - gamma
    The input dictionaries define the min/max/number of steps for each of the
    coordinates.
    The profile will be computed for mass=1, and with the appropriate value
    of rho_s to be properly normalized (i.e., u(k) --> 1 as k --> 0).
    We also save an array containing the computed rho_s (for mass=1,
    rvir=1) as function of log10(concentration) and gamma for later use.
    As this should be called only once before the runs, I use the
    slower (but more precise) function 'alt_rhos_modNFW' to compute rho_s.
    """

    k_rvir = np.logspace(log_kvals_rvir_dict['min'],
                         log_kvals_rvir_dict['max'],
                         log_kvals_rvir_dict['N'])
    log_k_rvir = np.log10(k_rvir)

    concentration = np.logspace(log_conc_dict['min'],
                                log_conc_dict['max'],
                                log_conc_dict['N'])
    log_conc = np.log10(concentration)

    gamma_exp = np.linspace(gamma_dict['min'],
                            gamma_dict['max'],
                            gamma_dict['N'])

    ft_hankel = hankel.SymmetricFourierTransform(ndim=3, N=hankelN,
                                                 h=hankelh)

    profile_grid = np.empty((log_kvals_rvir_dict['N'], log_conc_dict['N'],
                             gamma_dict['N']))

    rho_s_grid = np.empty((log_conc_dict['N'], gamma_dict['N']))

    for i, conc in enumerate(concentration):
        if verbose:
            print "Concentration = %f (%d of %d)" % (conc, i,
                                                     log_conc_dict['N'])
        for j, g_exp in enumerate(gamma_exp):

            rho_s = alt_rhos_modNFW(mass=1, rvir=1, conc=conc, gamma=g_exp)
            norm_prof_func = \
                lambda x: profile_ModNFW_config_scalar(rvals=x, rho_s=1,
                                                       conc=conc, rvir=1,
                                                       gamma_exp=g_exp)
            profile_grid[:, i, j] = \
                rho_s*ft_hankel.transform(f=norm_prof_func, k=k_rvir,
                                          ret_err=False, ret_cumsum=False)
            rho_s_grid[i, j] = rho_s

    np.savez(output_file, log10_k_rvir=log_k_rvir,
             log10_concentration=log_conc, gamma=gamma_exp,
             profile_grid=profile_grid, rho_s_unit=rho_s_grid)

    log_file = output_file + ".log"
    with open(log_file, 'a') as flog:
        flog.write("Arrays in " + output_file + " created by function "
                   "densprofile.create_profile_grid_fourier(), with options:\n")
        flog.write("Limits for log(k * rvir) array: " +
                   str(log_kvals_rvir_dict) + "\n")
        flog.write("Limits for log(concentration) array: " +
                   str(log_conc_dict) + "\n")
        flog.write("Limits for gamma array: " + str(gamma_dict) + "\n")
        flog.write("Hankel parameters: N = %d, h = %f\n" % (hankelN, hankelh))

    return 0


def profile_ModNFW_fourier_from_grid(kvals, mass, rho_s, rvir, conc,
                                     gamma=1.0, log_krvir_grid=None,
                                     log_conc_grid=None, gamma_grid=None,
                                     profile_fourier_grid=None):
    """
    Function to compute the Fourier-space profile for the ModNFW model,
    based on interpolation in a pre-computed grid.

    Will need to re-scale the k values, get the correct normalization of the
    profile.

    It will also make a cut so that u(k) = 1 for k < 1/(10 rvir).

    This function assumes mass, rho_s, rvir, conc are 1-D arrays of the same
    length, Nm.
    (However, values of rho_s, mass are not actually used, as we assume
    that the pre-computed profile is already correctly normalized).

    kvals is also a 1-D array of arbitrary length Nk.
    gamma is just a float (not an array)

    Returns an array of shape (Nk, Nm).
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1
    Nk = len(kvals)

    mass = np.atleast_1d(mass)
    rho_s = np.atleast_1d(rho_s)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rho_s) == len(rvir) == len(conc)

    # Check that the grid arrays make sense
    Nkgrid = len(log_krvir_grid)
    Ncgrid = len(log_conc_grid)
    Nggrid = len(gamma_grid)

    assert profile_fourier_grid.shape == (Nkgrid, Ncgrid, Nggrid)

    # Create the appropriate interpolator object, where we allow for linear
    # extrapolation (expect to have as input very small k values, that will
    # be solved by our cut)
    prof_3d_interpolator = \
        RegularGridInterpolator((log_krvir_grid, log_conc_grid, gamma_grid),
                                profile_fourier_grid,
                                bounds_error=False, fill_value=None)

    # Prepare the input coordinates for the interpolation
    coords_input = np.empty((Nk, Nm, 3))

    log_krvir_input = np.log10(np.outer(kvals, rvir))
    coords_input[:, :, 0] = log_krvir_input

    log_conc_input = np.outer(np.ones(Nk), np.log10(conc))
    coords_input[:, :, 1] = log_conc_input

    gamma_input = gamma*np.ones((Nk, Nm))
    coords_input[:, :, 2] = gamma_input

    # Do the interpolation to get the result
    normed_uprofile = prof_3d_interpolator(coords_input)

    # Do our cut at small k
    normed_uprofile[log_krvir_input < -1] = 1

    return normed_uprofile


def rhos_ModNFW_from_grid(mass=1e10, rvir=1.0, conc=10.0, gamma=1.0,
                          log_conc_grid=None, gamma_grid=None,
                          rho_s_unit_grid=None):
    """
    Function to obtain the normalization parameter rho_s for a ModNFW
    profile by interpolation from a pre-computed grid of 'unit'
    rho_s as function of concentration and gamma, as obtained from
    'create_profile_grid_fourier()'.
    We do linear interpolation over 'log(rho_s)' instead of 'rho_s' as we
    expect this to give better results given the typical dependence of
    rho_s on concentration
    """

    mass = np.atleast_1d(mass)
    rvir = np.atleast_1d(rvir)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rvir.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rvir) == len(conc)

    # Check that the grid arrays make sense
    Ncgrid = len(log_conc_grid)
    Nggrid = len(gamma_grid)

    assert rho_s_unit_grid.shape == (Ncgrid, Nggrid)

    # Create the appropriate interpolator object, where we allow for linear
    # extrapolation (although we do not expect to need this, hopefully
    # pre-computed grid in conc-gamma plane should cover all the needed
    # parameter space)
    rho_s_unit_2d_interpolator = \
        RegularGridInterpolator((log_conc_grid, gamma_grid),
                                np.log(rho_s_unit_grid),
                                bounds_error=False, fill_value=None)

    # Prepare the input coordinates for the interpolation
    coords_input = np.empty((Nm, 2))
    coords_input[:, 0] = np.log10(conc)
    coords_input[:, 1] = gamma*np.ones(Nm)

    # Do the interpolation to get the 'normalised' result
    rho_s_unit_out = np.exp(rho_s_unit_2d_interpolator(coords_input))

    # Re-normalise to get the desired normalisation for our values of
    # mass, rvir
    rho_s_out = rho_s_unit_out*mass*(rvir**(-3))

    return rho_s_out


class HaloProfileNFW(object):
    """Class that describes a Navarro-Frenk-White profile for a halo of a given
       mass, and for a given cosmology and redshift.
    """

    def __init__(self, mass=1e10, redshift=0, cosmo=ac.WMAP7,
                 powesp_lin_0=None, c_zero=11.0, beta=0.13,
                 logM_min=10.0, logM_max=16.0, logM_step=0.05):
        """
        Parameters defining the NFW halo profile:

        mass: mass of the halo (in M_sol) -- float or array of floats
        redshift -- float
        cosmo: an astropy.cosmology object defining the cosmology
        powesp_lin_0: a PowerSpectrum object containing the z=0 linear power
                      spectrum corresponding to this same cosmology
        c_zero, beta: parameters for the concentration relation.
                      Probably best to leave at the default values
        logM_min, logM_max, logM_step: parameters of the mass array used in
                      the calculation of M_star (needed for the concentration)

        Class adapted to work for an array of masses, not only a single value
        """

        # Convert input mass to array if it is not, and check it is only 1D!
        mass = np.atleast_1d(mass)
        assert mass.ndim == 1

        self.Nmass = len(mass)
        self.mass = mass
        self.cosmo = cosmo
        self.powesp_lin_0 = powesp_lin_0
        self.redshift = redshift

        # These parameters will now be also 1D arrays
        self.rvir = rvir_from_mass(mass=mass, redshift=redshift, cosmo=cosmo)
        self.conc = concentration(mass=mass, redshift=redshift, cosmo=cosmo,
                                  powesp_lin_0=powesp_lin_0,
                                  c_zero=c_zero, beta=beta,
                                  logM_min=logM_min, logM_max=logM_max,
                                  logM_step=logM_step)
        self.r_s = self.rvir/self.conc
        self.rho_s = rhos_from_charact(mass=self.mass, rvir=self.rvir,
                                       conc=self.conc)

    def profile_config(self, r):
        """Returns the halo density profile in configuration space,
           as function of the scale r (can be an array).
           From eq. (A8) in C2012.

           Returns an array of shape (Nr, Nmass), where Nr is the number
           of scales given as input.
        """

        return profile_NFW_config_parameters(r, self.rho_s, self.conc,
                                             self.rvir)

    def profile_config_norm(self, r):
        """
        Returns the *normalized* halo density profile in configuration space,
        as function of the scale r (can be an array).

        This normalized profile is the one needed for the calculation of the
        central-satellite term.

        Returns an array of shape (Nr, Nmass), where Nr si the number of scales
        given as input.
        """

        return self.profile_config(r)/self.mass

    def profile_fourier(self, k):
        """Returns the normalised halo density profile in Fourier space,
           as function of the wavenumber k (can be an array).
           From eq. (81) in CS02

           Returns an array of shape (Nk, Nmass), where Nk is the number
           of scales given as input.
        """

        return profile_NFW_fourier_parameters(k, self.mass, self.rho_s,
                                              self.rvir, self.conc)


class HaloProfileModNFW(HaloProfileNFW):
    """
    Class that describes a *modified* Navarro-Frenk-White profile for a halo
    of a given mass, with additional free parameters f_gal and gamma
    following the model of Watson et al. (2010).

    We inherit from the class corresponding to the standard NFW profile.
    """

    def __init__(self, mass=1e10, f_gal=1.0, gamma=1.0,
                 redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None,
                 c_zero=11.0, beta=0.13,
                 logM_min=10.0, logM_max=16.0, logM_step=0.05,
                 four_grid_log_krvir=None, four_grid_log_conc=None,
                 four_grid_gamma=None, four_grid_profile=None,
                 fourier_ft_hankel=None, fourier_Nk_interp=None,
                 fourier_Nm_interp=None):
        """
        Parameters defining the NFW halo profile:

        mass: mass of the halo (in M_sol) -- float or array of floats
        f_gal: relation between galaxy concentration and DM concentration
               For NFW, f_gal=1
        gamma: inner slope of the density profile
               For NFW, gamma=1
        redshift -- float
        cosmo: an astropy.cosmology object defining the cosmology
        powesp_lin_0: a PowerSpectrum object containing the z=0 linear power
                      spectrum corresponding to this same cosmology
        c_zero, beta: parameters for the concentration relation.
                      Probably best to leave at the default values
        logM_min, logM_max, logM_step: parameters of the mass array used in
                      the calculation of M_star (needed for the concentration)
        four_grid_log_krvir, four_grid_log_conc, four_grid_gamma,
        four_grid_profile: data corresponding to a pre-computed Fourier-space
            profile to be used to compute the required profile using
            interpolation.
            These should be the arrays needed by the
            profile_ModNFW_fourier_from_grid() function.
            If either of these is 'None', will actually calculate the profile,
            as described below.
        fourier_ft_hankel, fourier_Nk_interp, fourier_Nm_interp:
            Only used if fourier_grid_data==None.
            Parameters defining the way in which we calculate the Fourier-space
            profile (using Hankel+interpolation). Only needed for gamma!=1.
            If they are 'None', will use default values in the function above.

        Class adapted to work for an array of masses, not only a single value
        """

        # Call initialization from parent
        super(HaloProfileModNFW,
              self).__init__(mass=mass, redshift=redshift,
                             cosmo=cosmo, powesp_lin_0=powesp_lin_0,
                             c_zero=c_zero, beta=beta,
                             logM_min=logM_min, logM_max=logM_max,
                             logM_step=logM_step)

        # Add the galaxy concentration
        self.f_gal = f_gal
        self.conc_gal = self.conc*self.f_gal

        # And modified parameters that depend on the concentration
        self.r_s_gal = self.rvir/self.conc_gal

        # Add the gamma (inner slope)
        self.gamma = gamma

        # Add parameters needed for the Fourier-transform of the profile
        # Decide whether to use pre-computed grid or direct
        # calculation, and assign needed parameters in each case
        if (four_grid_log_krvir is not None) and \
           (four_grid_log_conc is not None) and\
           (four_grid_gamma is not None) and\
           (four_grid_profile is not None):

            self.fourier_use_grid = True
            self.four_grid_log_krvir = four_grid_log_krvir
            self.four_grid_log_conc = four_grid_log_conc
            self.four_grid_gamma = four_grid_gamma
            self.four_grid_profile = four_grid_profile

        else:
            self.fourier_use_grid = False
            self.fourier_ft_hankel = fourier_ft_hankel
            self.fourier_Nk_interp = fourier_Nk_interp
            self.fourier_Nm_interp = fourier_Nm_interp

        # Add modified parameters that depend on both conc and gamma
        if self.gamma == 1:
            self.rho_s_gal = rhos_from_charact(mass=self.mass,
                                               rvir=self.rvir,
                                               conc=self.conc_gal)
        else:
            self.rho_s_gal = rhos_from_charact_modNFW(mass=self.mass,
                                                      rvir=self.rvir,
                                                      conc=self.conc_gal,
                                                      gamma=self.gamma)

    def mod_profile_config(self, r):
        """
        Returns the halo *galaxy* density profile in configuration space,
        for the *modified* NFW case, as function of the scale r (can be an
        array).

        Returns an array of shape (Nr, Nmass), where Nr is the number of
        scales given as input.
        """

        if self.gamma == 1:
            return profile_NFW_config_parameters(r, self.rho_s_gal,
                                                 self.conc_gal, self.rvir)
        else:
            return profile_ModNFW_config_parameters(r, self.rho_s_gal,
                                                    self.conc_gal, self.rvir,
                                                    self.gamma)

    def mod_profile_config_norm(self, r):
        """
        Returns the *normalized* halo *galaxy* density profile in
        configuration space, for the *modified* NFW case, as function of
        the scale r (can be an array).

        This normalized profile is the one needed for the calculation of
        the central-satellite term.

        Returns an array of shape (Nr, Nmass), where Nr is the number of
        scales given as input.
        """

        return self.mod_profile_config(r)/self.mass

    def mod_profile_fourier(self, k):
        """
        Returns the normalised halo *galaxy* density profile in Fourier
        space, for the *modified* NFW case, as function of the wavenumber
        k (can be an array).

        Returns an array of shape (Nk, Nmass), where Nk is the number
        of scales given as input.
        """

        if self.gamma == 1:
            return profile_NFW_fourier_parameters(k, self.mass, self.rho_s_gal,
                                                  self.rvir, self.conc_gal)

        elif self.fourier_use_grid:
            return \
                profile_ModNFW_fourier_from_grid(k, self.mass, self.rho_s_gal,
                                                 self.rvir, self.conc_gal,
                                                 self.gamma,
                                                 log_krvir_grid=self.four_grid_log_krvir,
                                                 log_conc_grid=self.four_grid_log_conc,
                                                 gamma_grid=self.four_grid_gamma,
                                                 profile_fourier_grid=self.four_grid_profile)

        else:
            return \
                profile_ModNFW_fourier_hankel_interp(k, self.mass,
                                                     self.rho_s_gal, self.rvir,
                                                     self.conc_gal, self.gamma,
                                                     ft_hankel=self.fourier_ft_hankel,
                                                     Nk_interp=self.fourier_Nk_interp,
                                                     Nm_interp=self.fourier_Nm_interp)
