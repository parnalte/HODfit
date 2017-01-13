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
from scipy.interpolate import UnivariateSpline
import hankel

import halomodel
import utils
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


def profile_NFW_config_parameters(rvals, rho_s, r_s, rvir):
    """
    Function that returns the standard NFW profile in configuration space
    (as function of scale 'rvals'), given the basic parameters
    rho_s (normalization), r_s (characteristic scale) and rvir (virial radius).

    We truncate the resulting profile at r=rvir.

    Assume that rho_s and r_s are arrays of length Nm, corresponding to Nm
    different haloes of several masses.

    Returns an array of shape (Nr, Nm), where Nr is the length of rvals.
    """

    rvals = np.atleast_1d(rvals)
    assert rvals.ndim == 1

    rho_s = np.atleast_1d(rho_s)
    r_s = np.atleast_1d(r_s)
    assert rho_s.ndim == 1
    assert r_s.ndim == 1
    assert len(rho_s) == len(r_s)

    fact1 = np.outer(rvals, 1./r_s)
    fact2 = pow(1. + fact1, 2.0)
    rho_h = rho_s/(fact1*fact2)

    rvir_grid, rvals_grid = np.meshgrid(rvir, rvals)
    rho_h[rvals_grid > rvir_grid] = 0
    return rho_h


def profile_ModNFW_config_parameters(rvals, rho_s, r_s, rvir, gamma=1):
    """
    Function that returns the *modified* NFW profile in configuration space
    (as function of scale 'rvals'), given the basic parameters
    rho_s (normalization), r_s (characteristic scale), rvir (virial radius),
    and gamma (inner slope).

    We truncate the resulting profile at r=rvir.

    Assume that rho_s, r_s and rvir are arrays of length Nm, corresponding to
    Nm different haloes of several masses.

    Returns an array of shape (Nr, Nm), where Nr is the length of rvals.
    """

    rvals = np.atleast_1d(rvals)
    assert rvals.ndim == 1

    rho_s = np.atleast_1d(rho_s)
    r_s = np.atleast_1d(r_s)
    rvir = np.atleast_1d(rvir)
    assert rho_s.ndim == 1
    assert r_s.ndim == 1
    assert rvir.ndim == 1
    assert len(rho_s) == len(r_s) == len(rvir)

    r_ratios = np.outer(rvals, 1./r_s)
    fact1 = pow(r_ratios, gamma)
    fact2 = pow(1. + r_ratios, 3. - gamma)
    rho_h = rho_s/(fact1*fact2)

    rvir_grid, rvals_grid = np.meshgrid(rvir, rvals)
    rho_h[rvals_grid > rvir_grid] = 0
    return rho_h


def integrand_unnorm(r, r_s, rvir, gamma):
    """
    Integrand function needed for the integral performed in
    rhos_from_charact_modNFW , to obtain the mass corresponding to an
    unnormalized ModNFW profile.

    Returns an array of shape (Nm, Nr).
    """
    r_s = np.atleast_1d(r_s)
    rvir = np.atleast_1d(rvir)
    assert r_s.ndim == 1
    assert rvir.ndim == 1
    Nm = len(r_s)
    assert Nm == len(rvir)

    rho_u = profile_ModNFW_config_parameters(rvals=r, rho_s=np.ones(Nm),
                                             r_s=r_s, rvir=rvir, gamma=gamma)
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

    r_s = rvir/conc

    # Define r-array to do the numerical (Simpson) integration
    # TODO: add options to define this array, and to test the results
    # For now, using rmin=1e-5, and Nr=1000, I get that the maximum relative
    # error in rho_s for gamma=1 (where I can compare to analytic formula)
    # is 5e-3
    max_rvir = rvir.max()
    r_vals_array = np.logspace(-5, np.log10(max_rvir), 1000)

    # Do numerical integral and get the normalization
    integ_unnorm_vals = integrand_unnorm(r=r_vals_array, r_s=r_s, rvir=rvir,
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
    r_s = rvir/conc

    integral_unnorm_m = np.empty(Nm)

    for i in range(Nm):
        integral_unnorm_m[i] = integrate.quad(integrand_unnorm,
                                              a=0, b=rvir[i],
                                              args=(r_s[i], rvir[i], gamma))[0]

    return mass/integral_unnorm_m


def profile_NFW_fourier_parameters(kvals, mass, rho_s, r_s, conc):
    """
    Function that returns the standard NFW profile in Fourier space
    (as function of wavenumber 'kvals'), given the basic parameters
    of the haloes.

    We take this from eq. (81) in CS02, so it already takes into account
    the truncation of the halo.

    Assume that mass, rho_s, r_s and conc are arrays of length Nm,
    corresponding to Nm different haloes of several masses.

    Returns an array of shape (Nk, Nm), where Nk is the length of kvals.
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1

    mass = np.atleast_1d(mass)
    rho_s = np.atleast_1d(rho_s)
    r_s = np.atleast_1d(r_s)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert r_s.ndim == 1
    assert conc.ndim == 1
    assert len(mass) == len(rho_s) == len(r_s) == len(conc)

    # Need to compute the sine and cosine integrals
    si_ckr, ci_ckr = spc.sici(np.outer(kvals, (1. + conc) * r_s))
    si_kr, ci_kr = spc.sici(np.outer(kvals, r_s))

    fact1 = np.sin(np.outer(kvals, r_s)) * (si_ckr - si_kr)
    fact2 = np.sin(np.outer(kvals, conc * r_s)) / \
        (np.outer(kvals, (1. + conc) * r_s))
    fact3 = np.cos(np.outer(kvals, r_s)) * (ci_ckr - ci_kr)

    uprof = 4.*np.pi*(rho_s / mass) * pow(r_s, 3.) * (fact1 - fact2 + fact3)

    return uprof


def integrand_fourier_transf(kvals, rvals, mass, rho_s, r_s, rvir, gamma=1.0):
    """
    Function that computes the integrand needed to compute the Fourier
    transform of the radial ModNFW profile.

    Assumes:
        kvals: length Nk
        rvals: length Nr
        mass, rho_s, r_s, rvir: scalars!!

    Trying to do this with vectors for mass, etc. would probably kill off the
    memory!!

    Returns array of shape (Nk, Nr)
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1

    rvals = np.atleast_1d(rvals)
    assert rvals.ndim == 1

    # Build the different terms:
    kr_term = np.outer(kvals, rvals)              # Shape (Nk,Nr)
    sinc_term = np.sin(kr_term)/kr_term           # (Nk, Nr)
    radial_term = 4.*np.pi*rvals*rvals*sinc_term  # (Nk, Nr)

    profile_term = profile_ModNFW_config_parameters(rvals, rho_s, r_s, rvir,
                                                    gamma)   # (Nr, 1)
    norm_profile_term = (profile_term/mass)[:, 0]  # Convert to 1D vector (Nr)

    # Now, we need to combine the radial_term and the norm_profile_term
    return radial_term*norm_profile_term


def profile_ModNFW_fourier_parameters(kvals, mass, rho_s, r_s, conc,
                                      gamma=1.0):
    """
    Function that returns the *modified* NFW profile in Fourier space
    (as function of wavenumber 'kvals'), given the basic parameters of the
    haloes.

    We do this using an explicit numerical integration, as there is no
    analytic result for gamma != 1.

    Assume that mass, rho_s, r_s and conc are arrays of length Nm,
    corresponding to Nm different haloes of several masses.

    Returns an array of shape (Nk, Nm), where Nk is the length of kvals.

    TODO: Check accuracy and speed of this approach and explore alternatives
          if needed. For now, just use a straightforward approach which will
          probably be very slow.
    """

    kvals = np.atleast_1d(kvals)
    assert kvals.ndim == 1
    Nk = len(kvals)

    mass = np.atleast_1d(mass)
    rho_s = np.atleast_1d(rho_s)
    r_s = np.atleast_1d(r_s)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert r_s.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rho_s) == len(r_s) == len(conc)

    rvir = r_s*conc

    # Define r-array to do the numerical (Simpson) integration
    # As we do explicit loop over mass, will 'adapt' it to the
    # rvir in each case, but define here the minimum value and no. of steps
    # TODO: add options to define this array, and to test the results
    logrvals_min = -5
    logrvals_step = 1e-3
    max_rvir = rvir.max()
    logrvals = np.arange(logrvals_min, np.log10(max_rvir), logrvals_step)
    rvals = 10**logrvals

    uprof_out = np.empty((Nk, Nm), float)
    for i in range(Nm):
        # rvals = np.logspace(logrvals_min, np.log10(rvir[i]), logrvals_step)
        integrand_vals = integrand_fourier_transf(kvals, rvals, mass[i],
                                                  rho_s[i], r_s[i], gamma)
        uprof_out[:, i] = integrate.simps(y=integrand_vals, x=rvals,
                                         even='first')

    return uprof_out


def profile_ModNFW_fourier_hankel(kvals, mass, rho_s, r_s, conc,
                                  gamma=1.0):
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
    r_s = np.atleast_1d(r_s)
    conc = np.atleast_1d(conc)
    assert mass.ndim == 1
    assert rho_s.ndim == 1
    assert r_s.ndim == 1
    assert conc.ndim == 1
    Nm = len(mass)
    assert Nm == len(rho_s) == len(r_s) == len(conc)

    rvir = r_s*conc
    sph_hankel = hankel.SphericalHankelTransform(nu=0, N=6000,
                                                 h=1e-5)
    uprof_out = np.empty((Nk, Nm), float)
    for i in range(Nm):
        norm_prof_func = lambda x: \
            profile_ModNFW_config_parameters(rvals=x, rho_s=rho_s[i],
                                             r_s=r_s[i], rvir=rvir[i],
                                             gamma=gamma)[:, 0]/mass[i]
        uprof_out[:, i] = utils.xir2pk_hankel(kvalues=kvals,
                                              xirfunction=norm_prof_func,
                                              sph_hankel=sph_hankel)
    # I add a condition, so that I make u(k)=1 for
    # all k < 1/(10 rvir).
    # This is a bit arbitrary, what works to avoid the oscillations
    # that appear at small k
    rvir_grid, kvals_grid = np.meshgrid(rvir, kvals)
    uprof_out[kvals_grid < 1./(10*rvir_grid)] = 1.0

    return uprof_out


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

        return profile_NFW_config_parameters(r, self.rho_s, self.r_s,
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
                                              self.r_s, self.conc)


class HaloProfileModNFW(HaloProfileNFW):
    """
    Class that describes a *modified* Navarro-Frenk-White profile for a halo
    of a given mass, with additional free parameters f_gal and gamma
    following the model of Watson et al. (2010).

    For now, only the addition of f_gal is implemented.

    We inherit from the class corresponding to the standard NFW profile.
    """

    def __init__(self, mass=1e10, f_gal=1.0, gamma=1.0,
                 redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None,
                 c_zero=11.0, beta=0.13,
                 logM_min=10.0, logM_max=16.0, logM_step=0.05):
        """
        Parameters defining the NFW halo profile:

        mass: mass of the halo (in M_sol) -- float or array of floats
        f_gal: relation between galaxy concentration and DM concentration
               For NFW, f_gal=1
        gamma (NOT implemented): inner slope of the density profile
               For NFW, gamma=1
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
                                                 self.r_s_gal, self.rvir)
        else:
            return profile_ModNFW_config_parameters(r, self.rho_s_gal,
                                                    self.r_s_gal, self.rvir,
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

        return profile_NFW_fourier_parameters(k, self.mass, self.rho_s_gal,
                                              self.r_s_gal, self.conc_gal)
