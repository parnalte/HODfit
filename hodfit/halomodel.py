
"""
hods_halomodel.py --- Halo model-related function/classes for the simple
                      HOD model

Author: P. Arnalte-Mur (ICC-Durham)
Creation date: 19/02/2014
Last modified: ----

This module will contain the classes and functions related to the halo model
(i.e. halo mass functions, halo bias, etc.)

Units: all masses will be in M_sol, and all distances in Mpc/h.

References in comments:
    C2012: Coupon et al. (2012) A&A, 542:A5
    MW02:  Mo & White (2002) MNRAS, 336:112


TODO:
   (Possibly): compute integrals over mass using Gaussian quadrature
       (Scipy's 'quad'), so we avoid the need to (arbitrarily) define a
       spacing in logM. This would only work if we transform ndens_diff_m
       so that we can integrate in d(logM), instead of d(M).
"""


import numpy as np
import astropy.cosmology as ac
from scipy import integrate
from scipy import misc

from utils import RHO_CRIT_UNITS


def delta_c_z(redshift=0, cosmo=ac.WMAP7):
    """Computes the linear critical density delta_c for a given cosmology
       and redshift. Computed from equation (A.3) in C2012.

       The function works correctly when 'redshift' is an array.
    """
    omz = cosmo.Om(redshift)
    result = (3./20.)*pow(12.*np.pi, 2./3.)*(1. + (0.013*np.log10(omz)))
    return result


def gz(redshift=0, cosmo=ac.WMAP7):
    """Auxiliar function needed to compute the linear growth factor.
       Defined as in eq. (10) of MW02.

       The function works correctly when 'redshift' is an array.
    """

    omz = cosmo.Om(redshift)
    olz = cosmo.Ode(redshift)
    denominator = pow(omz, 4./7.) - olz + ((1. + (omz/2.))*(1. + (olz/70.)))
    return (5./2.)*omz/denominator


def growth_factor_linear(redshift=0, cosmo=ac.WMAP7):
    """Computes the growth factor for linear fluctuations, D(z), as defined
       by eq.(10) in MW02.

       The function works correctly when 'redshift' is an array.
    """

    gval_z = gz(redshift=redshift, cosmo=cosmo)
    gval_0 = gz(redshift=0, cosmo=cosmo)

    return gval_z/(gval_0*(1. + redshift))


def R_rms(mass=1e10, cosmo=ac.WMAP7):
    """Computes the radius of the filter adequate for a given mass
       to compute rms fluctuation in eq. (A5) of C2012.
       Input mass in M_sol, output radius in Mpc/h.

       The function works correctly when 'redshift' is an array.
    """

    mass_dens = RHO_CRIT_UNITS*cosmo.Om0

    return pow(3.*mass/(4.*np.pi*mass_dens), 1./3.)


def w_tophat_fourier(x):
    """Auxiliar function for spherical top-hat filter in Fourier space
    """

    return (3./(x**3))*(np.sin(x) - (x*np.cos(x)))


def sigma_radius(radius=8.0, powesp=None):
    """Computes the rms density fluctuations in a top-hat filter of width radius
       computed from the given power spectrum. From eq. (A5) in C2012.
       Length units should be consistent between radius and powesp.

       Adapted to work correctly when 'radius' is an array.
    """

    # Convert input to array if it is not, and check it is only 1D!
    radius = np.atleast_1d(radius)
    assert radius.ndim == 1

    wvalues = w_tophat_fourier(np.outer(radius, powesp.k))
    integrand_array = (powesp.k**2)*powesp.pk*(wvalues**2)

    integ_result = integrate.simps(x=powesp.k, y=integrand_array, axis=1,
                                   even='first')

    sigma = np.sqrt(integ_result/(2.*(np.pi**2)))

    return sigma


def sigma_mass(mass=1e10, cosmo=ac.WMAP7, powesp_lin_0=None):
    """Compute the rms density fluctuation corresponding to a given mass,
       following eq. (A5) in C2012.
       Units: masses in M_sol, distances (and powesp) in Mpc/h
       The power spectrum should be the z=0, linear power spectrum
       corresponding to the Cosmology 'cosmo' used.

       The function works correctly when 'redshift' is an array.
    """

    # First, get the corresponding radius of the filter
    R = R_rms(mass=mass, cosmo=cosmo)

    # And now, get sigma
    return sigma_radius(radius=R, powesp=powesp_lin_0)


def nu_variable(mass=1e10, redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """Make the conversion from halo mass to the 'nu' variable,
       for a given cosmology and redshift.
       This is from eq. (A2) in C2012.
       Units: masses in M_sol, distances (and powesp) in Mpc/h
       The power spectrum should be the z=0, linear power spectrum
       corresponding to the Cosmology 'cosmo' used.

       The function works correctly when either 'mass' OR 'redshift' are an
       array, but NOT when BOTH inputs are arrays.
    """

    dc = delta_c_z(redshift=redshift, cosmo=cosmo)
    Dz = growth_factor_linear(redshift=redshift, cosmo=cosmo)
    sig = sigma_mass(mass=mass, cosmo=cosmo, powesp_lin_0=powesp_lin_0)

    return dc/(Dz*sig)


class HaloModelMW02():
    """Class that contains all the functions related to the halo model defined
       in MW02 (in particular eqs. 14,19), for a given cosmology
       (and parameters).
    """

    def __init__(self, cosmo=ac.WMAP7, powesp_lin_0=None, redshift=0,
                 par_Amp=0.322, par_a=1./np.sqrt(2.),
                 par_b=0.5, par_c=0.6, par_q=0.3):
        """Parameters defining the Halo Model:

           cosmo: an astropy.cosmology object defining the cosmology
           powesp_lin_0: a PowerSpectrum object containing the z=0 linear
               power spectrum corresponding to this same cosmology
           redshift: redshift at which we do the calculations
           par_Amp, par_a, par_b, par_c, par_q: parameters of the model.
               Typically it's best to leave them at the defaults
        """

        self.cosmo = cosmo
        self.powesp_lin_0 = powesp_lin_0
        self.redshift = redshift
        self.par_Amp = par_Amp
        self.par_a = par_a
        self.par_b = par_b
        self.par_c = par_c
        self.par_q = par_q

        # Mass-dependent arrays we will eventually use
        # Will need to be initialized elsewhere
        self.mass_array = None
        self.Nm = 0
        self.nuvar_array = None
        self.bias_array = None
        self.ndens_diff_m_array = None

    def nu_variable(self, mass=1e12):
        """Make the conversion from halo mass to the 'nu' variable,
           for a given cosmology and redshift.
           Wrapper over external function 'nu_variable'.

           This function works correctly when 'mass' is an array.
        """

        return nu_variable(mass=mass, redshift=self.redshift, cosmo=self.cosmo,
                           powesp_lin_0=self.powesp_lin_0)

    def bias_nu(self, nuval):
        """Bias function (as function of nu) defined in eq. (19) of MW02

           This function works correctly when 'nuval' is an array.
        """

        nu_alt = np.sqrt(self.par_a)*nuval
        dc = delta_c_z(redshift=self.redshift, cosmo=self.cosmo)

        term1 = nu_alt**2.
        term2 = self.par_b*pow(nu_alt, 2.*(1. - self.par_c))
        denominator = pow(nu_alt, 2.*self.par_c) +\
            (self.par_b*(1. - self.par_c)*(1. - (0.5*self.par_c)))
        term3 = (pow(nu_alt, 2.*self.par_c)/np.sqrt(self.par_a))/denominator

        bias = 1. + ((term1 + term2 - term3)/dc)

        return bias

    def bias_fmass(self, mass=1e12):
        """Bias function for haloes of a fixed mass, from above

           This function works correctly when 'mass' is an array.
        """

        nuval = self.nu_variable(mass=mass)

        return self.bias_nu(nuval)

    def ndens_differential(self, mass=1e12):
        """Calculates the 'differential' part of eq. (14) in MW02.
           Note there's a typo in this equation: the mean mass density
           has to be at z=0!
           In order to get the appropriate integral terms, should integrate
           ndens_differential*dnu

           This function works correctly when 'mass' is an array.
        """

        nuval = self.nu_variable(mass=mass)
        nu_alt = np.sqrt(self.par_a)*nuval

        rho_mean_present = self.cosmo.Om0*RHO_CRIT_UNITS

        # The sqrt(a) term comes from the d(nu') term, so that
        # we can do the actual integral over nu
        term1 = np.sqrt(self.par_a)*self.par_Amp*np.sqrt(2./np.pi)
        term2 = 1. + (1./pow(nu_alt, 2*self.par_q))
        term3 = rho_mean_present/mass
        term4 = np.exp(-nu_alt*nu_alt/2.)

        return term1*term2*term3*term4

    def ndens_diff_m(self, mass=1e12, delta_m_rel=1e-4):
        """
        Calculates the 'differential' part of eq. (14) in MW02 including the
        nu-to-mass transform. In this way, now the integral terms can be
        obtained integrating directly ndens_diff_m*dM

        This will be just a wrapper over ndens_differential() adding the
        differential term dnu/dM.

        The delta_m_rel parameter sets the relative spacing to be used in the
        finite differences approximation of the derivative.

        This function works well when 'mass' is an array.
        """

        dnudM = misc.derivative(func=self.nu_variable, x0=mass,
                                dx=delta_m_rel*mass, n=1)

        return dnudM*self.ndens_differential(mass=mass)

    def ndens_integral(self, logM_min=10.0, logM_max=16.0, reltol=1e-5):
        """
        Compute the total number density of haloes given a range in halo
        mass.

        Define the mass range in log10 space, units of M_sol.

        Use Romberg integration (from Scipy), to the relative tolerance set
        by the parameter 'reltol'.
        We do a change of variable to integrate over x=log10(M).
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert reltol > 0

        def integrand(x):
            return np.log(10)*(10**x)*self.ndens_diff_m(mass=10**x)

        int_result = integrate.romberg(integrand, a=logM_min, b=logM_max,
                                       tol=0, rtol=reltol, vec_func=True)
        return int_result[0]

    def mean_bias(self, logM_min=10.0, logM_max=16.0, reltol=1e-5):
        """
        Compute the mean halo bias given a range in halo mass.

        Define the mass ranges in log10 space, units of M_sol.

        Use Romberg integration (from Scipy), to the relative tolerance set
        by the parameter 'reltol'.
        We do a change of variable to integrate over x=log10(M).
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert reltol > 0

        def integrand(x):
            return np.log(10)*(10**x)*self.bias_fmass(mass=10**x) * \
                self.ndens_diff_m(mass=10**x)

        int_result = integrate.romberg(integrand, a=logM_min, b=logM_max,
                                       tol=0, rtol=reltol, vec_func=True)

        return int_result[0] / \
            self.ndens_integral(logM_min, logM_max, reltol)

    def mean_mass(self, logM_min=10.0, logM_max=16.0, reltol=1e-5):
        """
        Compute the mean halo mass given a range in halo mass.

        Define the mass ranges in log10 space, units of M_sol.

        Use Romberg integration (from Scipy), to the relative tolerance set
        by the parameter 'reltol'.
        We do a change of variable to integrate over x=log10(M).
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert reltol > 0

        def integrand(x):
            return np.log(10)*(10**(2*x))*self.ndens_diff_m(mass=10**x)

        int_result = integrate.romberg(integrand, a=logM_min, b=logM_max,
                                       tol=0, rtol=reltol, vec_func=True)

        return int_result[0] / \
            self.ndens_integral(logM_min, logM_max, reltol)

    def integral_quantities(self, logM_min=10.0, logM_max=16.0,
                            reltol=1e-5):
        """
        For a given range of masses, compute the interesting integral
        quantities doing an integral over the differential mass function:
        This function returns:
          - Integral mass function: number density of haloes for the mass range
            (identical to the result of 'ndens_integral_quad')
          - Mean bias for the given range (identical to 'mean_bias_quad')
          - Mean halo mass for the given range (identical to 'mean_mass_quad')

        Define the mass ranges in log10 space, units of M_sol.

        Use Romberg integration (from Scipy), to the relative tolerance set
        by the parameter 'reltol'.
        We do a change of variable to integrate over x=log10(M).
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert reltol > 0

        def integrand_ndens(x):
            return np.log(10)*(10**x)*self.ndens_diff_m(mass=10**x)

        def integrand_bias(x):
            return np.log(10)*(10**x)*self.bias_fmass(mass=10**x) * \
                self.ndens_diff_m(mass=10**x)

        def integrand_mass(x):
            return np.log(10)*(10**(2*x))*self.ndens_diff_m(mass=10**x)

        ndens_int = integrate.romberg(integrand_ndens, a=logM_min, b=logM_max,
                                      tol=0, rtol=reltol, vec_func=True)[0]

        mean_bias = integrate.romberg(integrand_bias, a=logM_min, b=logM_max,
                                      tol=0, rtol=reltol,
                                      vec_func=True)[0]/ndens_int

        mean_mass = integrate.romberg(integrand_mass, a=logM_min, b=logM_max,
                                      tol=0, rtol=reltol,
                                      vec_func=True)[0]/ndens_int

        return ndens_int, mean_bias, mean_mass

    def set_mass_arrays(self, logM_min=10, logM_max=16, logM_step=0.05,
                        delta_m_rel=1e-4):
        """
        Compute and store the relevant mass-dependent quantities for a given
        mass array, so we can re-use without the need to re-calculating them
        each time
        """

        # Check the mass array makes sense
        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        self.mass_array = 10**np.arange(logM_min, logM_max, logM_step)
        self.Nm = len(self.mass_array)

        self.nuvar_array = self.nu_variable(mass=self.mass_array)
        self.bias_array = self.bias_nu(nuval=self.nuvar_array)
        self.ndens_diff_m_array = \
            self.ndens_diff_m(mass=self.mass_array, delta_m_rel=delta_m_rel)
