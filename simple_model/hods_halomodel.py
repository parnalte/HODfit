
"""
   hods_halomodel.py --- Halo model-related function/classes for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: 19/02/2014

   This module will contain the classes and functions related to the halo model
   (i.e. halo mass functions, halo bias, etc.)

   Units: all masses will be in M_sol, and all distances in Mpc/h.

   References in comments:
       C2012: Coupon et al. (2012) A&A, 542:A5
       MW02:  Mo & White (2002) MNRAS, 336:112
"""


import numpy as np
import astropy.cosmology as ac
from hods_utils import RHO_CRIT_UNITS, PowerSpectrum




def delta_c_z(redshift=0, cosmo=ac.WMAP7):
    """Computes the linear critical density delta_c for a given cosmology
       and redshift. Computed from equation (A.3) in C2012
    """
    omz = cosmo.Om(redshift)
    result = (3./20.)*pow(12.*np.pi, 2./3.)*(1. + (0.013*np.log10(omz)))
    return result


def gz(redshift=0, cosmo=ac.WMAP7):
    """Auxiliar function needed to compute the linear growth factor.
       Defined as in eq. (10) of MW02
    """

    omz = cosmo.Om(redshift)
    olz = cosmo.Ode(redshift)
    denominator = pow(omz, 4./7.) - olz + ((1. + (omz/2.))*(1. + (olz/70.)))
    return (5./2.)*omz/denominator

    
def growth_factor_linear(redshift=0, cosmo=ac.WMAP7):
    """Computes the growth factor for linear fluctuations, D(z), as defined
       by eq.(10) in MW02
    """

    gval_z = gz(redshift=redshift, cosmo=cosmo)
    gval_0 = gz(redshift=0, cosmo=cosmo)

    return gval_z/(gval_0*(1. + redshift))


def R_rms(mass=1e10, cosmo=ac.WMAP7):
    """Computes the radius of the filter adequate for a given mass
       to compute rms fluctuation in eq. (A5) of C2012
       Input mass in M_sol, output radius in Mpc/h
    """

    mass_dens = RHO_CRIT_UNITS*cosmo.Om0

    return pow(3.*mass/(4.*np.pi*mass_dens), 1./3.)

    
def w_tophat_fourier(x):
    """Auxiliar function for spherical top-hat filter in Fourier space
    """

    return (3./pow(x,3))*(np.sin(x) - (x*np.cos(x)))



def sigma_radius(radius = 8.0, powesp=None):
    """Computes the rms density fluctuations in a top-hat filter of width radius
       computed from the given power spectrum. From eq. (A5) in C2012.
       Length units should be consistent between radius and powesp.
    """

    #We do the integral using a simple trapezium rule given the k-values
    #at which the P(k) is sampled
    x2=powesp.k[1:]
    x1=powesp.k[:-1]

    
    wvalues = w_tophat_fourier(radius*powesp.k)
    integrand_array = (powesp.k**2)*powesp.pk*(wvalues**2)

    y2 = integrand_array[1:]
    y1 = integrand_array[:-1]
    comb_array = 0.5*(y2+y1)*(x2-x1)
    integ_result = comb_array.sum()

    sigma = np.sqrt(integ_result/(2.*(np.pi**2)))
    return sigma


def sigma_mass(mass = 1e10, cosmo = ac.WMAP7, powesp_lin_0 = None):
    """Compute the rms density fluctuation corresponding to a given mass,
       following eq. (A5) in C2012.
       Units: masses in M_sol, distances (and powesp) in Mpc/h
       The power spectrum should be the z=0, linear power spectrum corresponding
       to the Cosmology 'cosmo' used.
    """

    #First, get the corresponding radius of the filter
    R = R_rms(mass=mass, cosmo=cosmo)

    #And now, get sigma
    return sigma_radius(radius=R, powesp=powesp_lin_0)

def nu_variable(mass = 1e10, redshift = 0, cosmo = ac.WMAP7, powesp_lin_0=None):
    """Make the conversion from halo mass to the 'nu' variable,
       for a given cosmology and redshift.
       This is from eq. (A2) in C2012.
       Units: masses in M_sol, distances (and powesp) in Mpc/h
       The power spectrum should be the z=0, linear power spectrum corresponding
       to the Cosmology 'cosmo' used.
    """

    dc = delta_c_z(redshift=redshift, cosmo=cosmo)
    Dz = growth_factor_linear(redshift=redshift, cosmo=cosmo)
    sig = sigma_mass(mass=mass, cosmo=cosmo, powesp_lin_0=powesp_lin_0)

    return dc/(Dz*sig)


    
    

class HaloModelMW02():
    """Class that contains all the functions related to the halo model defined
       in MW02 (in particular eqs. 14,19), for a given cosmology (and parameters).
    """

    def __init__(self, cosmo=ac.WMAP7, powesp_lin_0=None, redshift=0, par_Amp=0.322, par_a=1./np.sqrt(2.),
                 par_b = 0.5, par_c = 0.6, par_q = 0.3):
        """Parameters defining the Halo Model:

           cosmo: an astropy.cosmology object defining the cosmology
           powesp_lin_0: a PowerSpectrum object containing the z=0 linear power spectrum corresponding to this same cosmology
           redshift: redshift at which we do the calculations
           par_Amp, par_a, par_b, par_c, par_q: parameters of the model. Typically it's best to leave them at the defaults
        """

        self.cosmo = cosmo
        self.powesp_lin_0 = powesp_lin_0
        self.redshift = redshift
        self.par_Amp = par_Amp
        self.par_a   = par_a
        self.par_b   = par_b
        self.par_c   = par_c
        self.par_q   = par_q



    def nu_variable(self, mass = 1e12):
        """Make the conversion from halo mass to the 'nu' variable,
        for a given cosmology and redshift.
        This is from eq. (A2) in C2012.
        Units: masses in M_sol, distances (and powesp) in Mpc/h
        The power spectrum should be the z=0, linear power spectrum corresponding
        to the Cosmology 'cosmo' used.
        """

        dc = delta_c_z(redshift=self.redshift, cosmo=self.cosmo)
        Dz = growth_factor_linear(redshift=self.redshift, cosmo=self.cosmo)
        sig = sigma_mass(mass=mass, cosmo=self.cosmo, powesp_lin_0=self.powesp_lin_0)

        return dc/(Dz*sig)


        
    def bias_nu(self, nuval):
        """Bias function (as function of nu) defined in eq. (19) of MW02
        """
        nu_alt = np.sqrt(self.par_a)*nuval
        dc = delta_c_z(redshift=self.redshift, cosmo=self.cosmo)

        term1 = nu_alt**2.
        term2 = self.par_b*pow(nu_alt, 2.*(1. - self.par_c))
        denominator = pow(nu_alt, 2.*self.par_c) + (self.par_b*(1. - self.par_c)*(1. - (0.5*self.par_c)))
        term3 = (pow(nu_alt, 2.*self.par_c)/np.sqrt(self.par_a))/denominator

        bias = 1. + ((term1 + term2 - term3)/dc)

        return bias


    def bias_fmass(self, mass = 1e12):
        """Bias function for haloes of a fixed mass, from above
        """

        nuval = self.nu_variable(mass=mass)

        return self.bias_nu(nuval)


    def ndens_differential(self, mass = 1e12):
        """Calculates the 'differential' part of eq. (14) in MW02.
           Note there's a typo in this equation: the mean mass density
           has to be at z=0!
           In order to get the appropriate integral terms, should integrate
           ndens_differential*dnu
        """

        nuval = self.nu_variable(mass=mass)
        nu_alt = np.sqrt(self.par_a)*nuval

        rho_mean_present = self.cosmo.Om0*RHO_CRIT_UNITS

        #The sqrt(a) term comes from the d(nu') term, so that
        #we can do the actual integral over nu
        term1 = np.sqrt(self.par_a)*self.par_Amp*np.sqrt(2./np.pi)
        term2 = 1. + (1./pow(nu_alt, 2*self.par_q))
        term3 = rho_mean_present/mass
        term4 = np.exp(-nu_alt*nu_alt/2.)

        return term1*term2*term3*term4


        
    def ndens_integral(self, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
        """Basically, just do the integral of the function above, given a range (and binning) in halo mass.
           This will give the mass function: number density of haloes in a given mass range.
           Define the mass ranges in log10 space, units of M_sol
        """

        assert logM_min > 0 
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)
        nsteps = len(mass_array)

        nbins = nsteps - 1

        sum_ndens = 0
        for i in range(nbins):

            M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
            nu_1 = self.nu_variable(mass=mass_array[i])
            nu_2 = self.nu_variable(mass=mass_array[i+1])

            sum_ndens = sum_ndens + (self.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

        return sum_ndens

    def integral_quantities(self, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
        """For a given range of masses (and integration steps), compute the interesting
           integral quantites doing an integral over the differential mass function:
           This function returns:
           - Integral mass function: number density of haloes for the mass range (identical
             to the result of 'ndens_integral')
           - Mean bias for the given range
           - Mean halo mass for the given range

        Define the mass ranges in log10 space, units of M_sol
        """
        
        assert logM_min > 0 
        assert logM_max > logM_min
        assert logM_step > 0

        mass_array = 10**np.arange(logM_min, logM_max, logM_step)
        nsteps = len(mass_array)

        nbins = nsteps - 1

        sum_ndens = 0.
        sum_bias = 0.
        sum_mass = 0.

        for i in range(nbins):

            M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
            nu_1 = self.nu_variable(mass=mass_array[i])
            nu_2 = self.nu_variable(mass=mass_array[i+1])
            bias_bin = self.bias_fmass(mass=M_mean)

            sum_ndens = sum_ndens + (self.ndens_differential(mass=M_mean)*(nu_2 - nu_1))
            sum_bias = sum_bias + (bias_bin*self.ndens_differential(mass=M_mean)*(nu_2 - nu_1))
            sum_mass = sum_mass + (M_mean*self.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

        bias_mean = sum_bias/sum_ndens
        mass_mean = sum_mass/sum_ndens

        return sum_ndens, bias_mean, mass_mean

    



        










        
        

    
    

    
    
    

    