"""
   hods_densprofile.py --- Profile-related function/classes for the simple HOD model

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
import hods_halomodel as hm
from hods_utils import RHO_CRIT_UNITS, PowerSpectrum


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

def rvir_from_mass(mass = 1e10, redshift=0, cosmo=ac.WMAP7):
    """Obtain the virial radius from the halo mass. From the 
       inversion of equation (A.11) in C2012.

       This function works OK when EITHER input 'mass' or 'redshift' are
       1D arrays, but NOT when BOTH of them are arrays.
    """
    

    rho_mean_0 = RHO_CRIT_UNITS*cosmo.Om0
    Dvir = Delta_vir(redshift=redshift, cosmo=cosmo)

    result = pow(3.*mass/(4.*np.pi*rho_mean_0*Dvir), 1./3.)

    return result


def mstar_interp(cosmo=ac.WMAP7, powesp_lin_0=None, logM_min = 10.0, logM_max  =16.0, logM_step = 0.05):
    """Obtain M_* from the condition sigma(M_*) = delta_c(z=0).
       Will get it from a linear interpolation to the function sigma(M).
    """

    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0
    
    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    #sigma_mass() already works well with input mass arrays
    sigma_array = hm.sigma_mass(mass=mass_array, cosmo=cosmo, powesp_lin_0=powesp_lin_0)

    delta_c0 = hm.delta_c_z(redshift=0, cosmo=cosmo)

    #We want to compute the function M(sigma) at the point sigma=delta_c0
    #First, need to sort the 'x' array, in this case sigma
    idx_sort = np.argsort(sigma_array)
    mass_star = np.interp(x=delta_c0, xp=sigma_array[idx_sort], fp=mass_array[idx_sort])

    return mass_star

def concentration(mass=1e10, redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None, c_zero=11.0, beta=0.13, logM_min = 10.0, logM_max  =16.0, logM_step = 0.05):
    """Concentration for a halo of a given mass, following eq. (A10)
       in C2012. Need mass array to obtain Mstar via interpolation

       This function works OK when EITHER input 'mass' or 'redshift' are
       1D arrays, but NOT when BOTH of them are arrays.
    """

    mass_star = mstar_interp(cosmo=cosmo, powesp_lin_0=powesp_lin_0, logM_min=logM_min, logM_max=logM_max, logM_step=logM_step)

    conc = (c_zero/(1. + redshift))*pow(mass/mass_star, -beta)

    return conc


def rhos_from_charact(mass=1e10, rvir=1.0, conc=10.0):
    """Obtain the normalization parameter, rho_s, given the characteristics of the profile
       (r_vir, concentration), and the total mass enclosed (mass).
       This follows from eq. (A9) in C2012.
    """

    term1 = 4.*np.pi*pow(rvir, 3)/pow(conc, 3)
    term2 = np.log(1. + conc) - (conc/(1.+conc))

    rho_s = mass/(term1*term2)

    return rho_s

    

class HaloProfileNFW():
    """Class that describes a Navarro-Frenk-White profile for a halo of a given
       mass, and for a given cosmology and redshift.
    """


    def __init__(self, mass=1e10, redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None, c_zero=11.0, beta=0.13,
                 logM_min = 10.0, logM_max  =16.0, logM_step = 0.05):
        """Parameters defining the NFW halo profile:

           mass: mass of the halo (in M_sol) -- float
           redshift -- float
           cosmo: an astropy.cosmology object defining the cosmology
           powesp_lin_0: a PowerSpectrum object containing the z=0 linear power spectrum corresponding
                         to this same cosmology
           c_zero, beta: parameters for the concentration relation. Probably best to leave at the default values
           logM_min, logM_max, logM_step: parameters of the mass array used in the calculation of M_star
           (needed for the concentration)
        """

        self.mass = mass
        self.cosmo = cosmo
        self.powesp_lin_0 = powesp_lin_0
        self.redshift = redshift

        self.rvir = rvir_from_mass(mass=mass, redshift=redshift, cosmo=cosmo)
        self.conc = concentration(mass=mass, redshift=redshift, cosmo=cosmo, powesp_lin_0=powesp_lin_0, c_zero=c_zero, beta=beta,
        logM_min=logM_min, logM_max=logM_max, logM_step=logM_step)
        self.r_s = self.rvir/self.conc
        self.rho_s = rhos_from_charact(mass=self.mass, rvir=self.rvir, conc=self.conc)


    def profile_config(self, r):
        """Returns the halo density profile in configuration space,
           as function of the scale r.
           From eq. (A8) in C2012
        """

        fact1 = r/self.r_s
        fact2 = pow(1. + fact1, 2.0)
        rho_h = self.rho_s/(fact1*fact2)
        return rho_h

    def profile_fourier(self, k):
        """Returns the normalised halo density profile in Fourier space,
           as function of the wavenumber k.
           From eq. (81) in CS02
        """

        #Need to compute the sine and cosine integrals
        si_ckr, ci_ckr = spc.sici((1.+self.conc)*k*self.r_s)
        si_kr, ci_kr   = spc.sici(k*self.r_s)

        fact1 = np.sin(k*self.r_s)*(si_ckr - si_kr)
        fact2 = np.sin(self.conc*k*self.r_s)/((1. + self.conc)*k*self.r_s)
        fact3 = np.cos(k*self.r_s)*(ci_ckr - ci_kr)

        uprof = 4.*np.pi*(self.rho_s/self.mass)*pow(self.r_s, 3.)*(fact1 - fact2 + fact3)

        return uprof


        