
"""
hods_halomodel.py --- Halo model-related function/classes for the simple
                      HOD model

Author: P. Arnalte-Mur (ICC-Durham)
Creation date: 19/02/2014
Last modified: 19/02/2014

This module will contain the classes and functions related to the halo model
(i.e. halo mass functions, halo bias, etc.)

Units: all masses will be in M_sol/h, and all distances in Mpc/h.

References in comments:
    C2012: Coupon et al. (2012) A&A, 542:A5
    MW02:  Mo & White (2002) MNRAS, 336:112
    Additional references in docstring for HaloModel class


TODO:
   (Possibly): compute integrals over mass using Gaussian quadrature
       (Scipy's 'quad'), so we avoid the need to (arbitrarily) define a
       spacing in logM. This would only work if we transform ndens_diff_m
       so that we can integrate in d(logM), instead of d(M).
TODO:
    Implement Tinker-2008 model for the HMF (seems standard today)
"""


import numpy as np
import astropy.cosmology as ac
from scipy import integrate
from scipy import misc

from .utils import RHO_CRIT_UNITS


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

def sigma_variable(mass=1e10, redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    Make the conversion from halo mass to the 'sigma' variable needed
    in some cases to model the HMF, for a given cosmology and redshift.
    This is from eq. (1) of Watson et al. (2013)

    Units: masses in M_sol, distances (and powesp) in Mpc/h
    The power spectrum should be the z=0, linear power spectrum
    corresponding to the Cosmology 'cosmo' used.
    """

    Dz = growth_factor_linear(redshift=redshift, cosmo=cosmo)
    sig_z0 = sigma_mass(mass=mass, cosmo=cosmo, powesp_lin_0=powesp_lin_0)

    return Dz*sig_z0


class HaloModel(object):
    """
        Class that contains the definition of the two main properties of the
        halo population:
        - the halo mass function (HMF)
        - the bias function as function of mass (BFM)

        We also include some mean quantities/properties (mean mass, mean bias, ...)
        that can be obtained directly from the above.

        We will implement different models for both the HMF and the BFM.
        Models implemented so far:
        * For the halo mass function (HMF)
            - Model from Sheth et al. (2001), as defined by eq. (14) of
              Mo&White (2002)
            - 'FOF-Universal' model from Watson et al. (2013), from their
              eq. (12)

        * For the bias as function of mass (BFM):
            - Model from Sheth et al. (2001), as defined by eq. (19) of
              Mo&White (2002)
            - Model from Tinker et al. (2005), from their appendix A.
              This is the same as the Sheth et al. (2001) model, but with
              updated parameter values
            - Model from Tinker et al. (2010), from their eq. (6) and Table 2.

        References:
        * Mo & White (2002), MNRAS, 336, 112-118
        * Sheth et al. (2001), MNRAS, 323, 1-12
        * Tinker et al. (2005), ApJ, 631, 41-58
        * Tinker et al. (2010), ApJ, 724, 878-886
        * Watson et al. (2013), MNRAS, 433, 1230-1245
    """

    def __init__(self, cosmo=ac.WMAP7, powesp_lin_0=None, redshift=0,
                 mass_function_model='Sheth2001',
                 bias_function_model='Sheth2001',
                 Delta=200.):
        """
        Parameters defining the Halo Model:

        cosmo: an astropy.cosmology object defining the cosmology
        powesp_lin_0: a PowerSpectrum object containing the z=0 linear
            power spectrum corresponding to this same cosmology
        redshift: redshift at which we do the calculations
        mass_function_model: string corresponding to the required HMF
            model. Implemented models are: ['Sheth2001']
        bias_function_model: string corresponding to the required BFM model.
            Implemented models are ['Sheth2001', 'Tinker2005', 'Tinker2010']
        Delta: overdensity used to define the halo population. Only used for
            certain models that use this as a parameter.
        """

        self.cosmo = cosmo
        self.powesp_lin_0 = powesp_lin_0
        self.redshift = redshift
        self.hmf_model = mass_function_model
        self.bfm_model = bias_function_model
        self.par_Delta = Delta

        # Check models and define needed parameters
        hmf_implemented_models = ['Sheth2001', 'Watson2013-FOF']
        bfm_implemented_models = ['Sheth2001', 'Tinker2005', 'Tinker2010', ]

        assert self.hmf_model in hmf_implemented_models, \
            f"Model {self.hmf_model} not implemented for HMF"
        assert self.bfm_model in bfm_implemented_models, \
            f"Model {self.bfm_model} not implemented for BFM"

        # Halo mass function
        if self.hmf_model == 'Sheth2001':
            self._hfm_formula = 'MoWhite2002'
            self._par_MW_Amp = 0.322
            self._par_MW_a = 1./np.sqrt(2.)
            self._par_MW_q = 0.3

        elif self.hmf_model == 'Watson2013-FOF':
            self._hfm_formula = 'Watson2013'
            self._par_W13_A = 0.282
            self._par_W13_alpha = 2.163
            self._par_W13_beta = 1.406
            self._par_W13_gamma = 1.210

        # Bias function
        if self.bfm_model == 'Sheth2001':
            self._bfm_formula = 'MoWhite2002'
            self._par_MW_a = 1./np.sqrt(2.)
            self._par_MW_b = 0.5
            self._par_MW_c  =0.6

        elif self.bfm_model == 'Tinker2005':
            self._bfm_formula = 'MoWhite2002'
            self._par_MW_a = 1./np.sqrt(2.)
            self._par_MW_b = 0.35
            self._par_MW_c = 0.8

        elif self.bfm_model == 'Tinker2010':
            self._bfm_formula = 'Tinker2010'
            # Define parameters from Table 2 of the paper
            y = np.log10(self.par_Delta)
            self._par_T10_A = 1.0 + 0.24*y*np.exp(-pow(4/y,4))
            self._par_T10_a = 0.44*y - 0.88
            self._par_T10_B = 0.183
            self._par_T10_b = 1.5
            self._par_T10_C = 0.019 + 0.107*y + 0.19*np.exp(-pow(4/y, 4))
            self._par_T10_c = 2.4


        # Mass-dependent arrays we will eventually use
        # Will need to be initialized elsewhere
        self.mass_array = None
        self.Nm = 0
        self.bias_array = None
        self.ndens_diff_m_array = None

    def _nu_variable(self, mass=1e12):
        """Make the conversion from halo mass to the 'nu' variable,
           for a given cosmology and redshift.
           Wrapper over external function 'nu_variable'.

           This function works correctly when 'mass' is an array.
        """

        return nu_variable(mass=mass, redshift=self.redshift, cosmo=self.cosmo,
                           powesp_lin_0=self.powesp_lin_0)

    def _sigma_variable(self, mass=1e12):
        """
        Make the conversion from halo mass to the 'sigma' variable used in
        certain HMF models.
        Wrapper over external function 'sigma_variable'.
        """

        return sigma_variable(mass=mass, redshift=self.redshift,
                              cosmo=self.cosmo,powesp_lin_0=self.powesp_lin_0)

    def _bias_nu_MW(self, nuval):
        """
        Bias function as function of nu defined in eq. (19) of Mo&White (2002)

        This function works correctly when 'nuval' is an array.
        """

        nu_alt = np.sqrt(self._par_MW_a)*nuval
        dc = delta_c_z(redshift=self.redshift, cosmo=self.cosmo)

        term1 = nu_alt**2.
        term2 = self._par_MW_b*pow(nu_alt, 2.*(1. - self._par_MW_c))
        denominator = pow(nu_alt, 2.*self._par_MW_c) +\
            (self._par_MW_b*(1. - self._par_MW_c)*(1. - (0.5*self._par_MW_c)))
        term3 = (pow(nu_alt, 2.*self._par_MW_c)/np.sqrt(self._par_MW_a))/denominator

        bias = 1. + ((term1 + term2 - term3)/dc)

        return bias

    def _bias_nu_T10(self, nuval):
        """
        Bias function as function of nu defined in eq. (6) of Tinker et al. (2010).

        """
        dc = delta_c_z(redshift=self.redshift, cosmo=self.cosmo)

        term_a = self._par_T10_A*(nuval**self._par_T10_a)/((nuval**self._par_T10_a) + (dc**self._par_T10_a))
        term_b = self._par_T10_B*(nuval**self._par_T10_b)
        term_c = self._par_T10_C*(nuval**self._par_T10_c)

        return 1. - term_a + term_b + term_c




    def bias_fmass(self, mass=1e12):
        """
        Bias function for haloes of a fixed mass, from the corresponding
        formula as function of the auxiliary variable nu/sigma/etc.
        (depending on model).

        This function works correctly when 'mass' is an array.
        """

        if self._bfm_formula == 'MoWhite2002':
            nuval = self._nu_variable(mass=mass)
            return self._bias_nu_MW(nuval)

        elif self._bfm_formula == 'Tinker2010':
            nuval = self._nu_variable(mass=mass)
            return self._bias_nu_T10(nuval)

    def _ndens_differential_MW(self, mass=1e12):
        """
        Calculates the 'differential' part of eq. (14) in Mo&White (2002).
         [Note there's a typo in this equation: the mean mass density
           has to be at z=0!]

        This is expressed as function of nu, so in order to get the appropriate
        integral terms, should integrate
            ndens_differential*dnu

        This function works correctly when 'mass' is an array.
        """

        nuval = self._nu_variable(mass=mass)
        nu_alt = np.sqrt(self._par_MW_a)*nuval

        rho_mean_present = self.cosmo.Om0*RHO_CRIT_UNITS

        # The sqrt(a) term comes from the d(nu') term, so that
        # we can do the actual integral over nu
        term1 = np.sqrt(self._par_MW_a)*self._par_MW_Amp*np.sqrt(2./np.pi)
        term2 = 1. + (1./pow(nu_alt, 2*self._par_MW_q))
        term3 = rho_mean_present/mass
        term4 = np.exp(-nu_alt*nu_alt/2.)

        return term1*term2*term3*term4

    def _f_sigma_W13(self, mass=1e12):
        """
        Calculates the 'halo multiplicity function' as function of the
        sigma variable according to eq. (12) in Watson et al. (2013)
        """

        sigmavar = self._sigma_variable(mass=mass)

        term1 = ((self._par_W13_beta/sigmavar)**self._par_W13_alpha) + 1.
        term2 = np.exp(-self._par_W13_gamma/(sigmavar**2))

        return self._par_W13_A*term1*term2

    def ndens_diff_m(self, mass=1e12, delta_m_rel=1e-4):
        """
        Calculates the 'differential' part of the halo mass function,
        already expressed in terms of mass, so that the integral terms
        can be obtained integrating directly ndens_diff_m*dM.

        This is obtained from the '_ndens_differential' or 'f' expressed
        in terms of the nu/sigma auxiliary variables adding the corresponding
        differential term dnu/dM or dsigma/dM and/or additional needed terms.

        The delta_m_rel parameter sets the relative spacing to be used in the
        finite differences approximation of the derivative.

        This function works well when 'mass' is an array.
        """

        if self._hfm_formula == 'MoWhite2002':
            dnudM = misc.derivative(func=self._nu_variable, x0=mass,
                                    dx=delta_m_rel*mass, n=1)
            return dnudM*self._ndens_differential_MW(mass=mass)

        elif self._hfm_formula == 'Watson2013':
            # Add terms from eq. (5) of the Watson paper
            rho_mean_present = self.cosmo.Om0*RHO_CRIT_UNITS

            def ln_invsigma_lnM(lnM):
                mass = np.exp(lnM)
                return np.log(1./self._sigma_variable(mass))

            d_lninvsigma_d_lnM = misc.derivative(func=ln_invsigma_lnM,
                                                 x0=np.log(mass),
                                                 dx=delta_m_rel*np.log(mass),
                                                 n=1)
            return rho_mean_present*self._f_sigma_W13(mass=mass)*d_lninvsigma_d_lnM/(mass**2)

    def ndens_integral(self, logM_min=10.0, logM_max=16.0, logM_step=0.05):
        """
        Computes the total number density of haloes given a range (and binning)
        in halo mass.

        Define the mass ranges in log10 space, units of M_sol.

        We will use simple Simpson integration rule, using Scipy's function
        for this.
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)

        nd_diff_array = self.ndens_diff_m(mass=mass_array)

        return integrate.simps(y=nd_diff_array, x=mass_array, even='first')

    def mean_bias(self, logM_min=10.0, logM_max=16.0, logM_step=0.05):
        """Compute the mean halo bias given a range (and binning) in halo mass.

           Define the mass ranges in log10 space, units of M_sol.

           We will integrate the bias over the mass function, using Scipy's
           function for Simpson's integration rule.
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)

        nd_diff_array = self.ndens_diff_m(mass=mass_array)
        bias_array = self.bias_fmass(mass=mass_array)

        return integrate.simps(y=(bias_array*nd_diff_array),
                               x=mass_array, even='first') /\
            self.ndens_integral(logM_min, logM_max, logM_step)

    def mean_mass(self, logM_min=10.0, logM_max=16.0, logM_step=0.05):
        """Compute the mean halo mass given a range (and binning) in halo mass.

           Define the mass ranges in log10 space, units of M_sol.

           We will integrate the mass over the mass function, using Scipy's
           function for Simpson's integration rule.
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)

        nd_diff_array = self.ndens_diff_m(mass=mass_array)

        return integrate.simps(y=(mass_array*nd_diff_array),
                               x=mass_array, even='first') /\
            self.ndens_integral(logM_min, logM_max, logM_step)

    def integral_quantities(self, logM_min=10.0, logM_max=16.0,
                            logM_step=0.05):
        """
        For a given range of masses (and integration steps), compute the
        interesting integral quantites doing an integral over the differential
        mass function:
        This function returns:
         - Integral mass function: number density of haloes for the mass
           range (identical to the result of 'ndens_integral')
         - Mean bias for the given range
         - Mean halo mass for the given range

        Define the mass ranges in log10 space, units of M_sol.

        Using SciPy's Simpson integration function
        """

        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)

        nd_diff_array = self.ndens_diff_m(mass=mass_array)
        bias_array = self.bias_fmass(mass=mass_array)

        ndens_integral = integrate.simps(y=nd_diff_array, x=mass_array,
                                         even='first')
        mean_bias = integrate.simps(y=(bias_array*nd_diff_array),
                                    x=mass_array, even='first')/ndens_integral
        mean_mass = integrate.simps(y=(mass_array*nd_diff_array),
                                    x=mass_array, even='first')/ndens_integral

        return ndens_integral, mean_bias, mean_mass

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

        self.bias_array = self.bias_fmass(mass=self.mass_array)
        self.ndens_diff_m_array = \
            self.ndens_diff_m(mass=self.mass_array, delta_m_rel=delta_m_rel)
