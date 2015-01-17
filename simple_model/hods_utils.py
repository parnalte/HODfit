"""
   hods_utils.py --- General utilities for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: ---

   TODO:
        - Change to a free (or my own) implementation of the Hankel transforms to get xi(r).
          An option is the library in https://pypi.python.org/pypi/hankel/0.1.0
          Other is to write it myself using functions in scipy.special
"""


import numpy as np
import astropy.cosmology as ac
import hankel
import contextlib
import sys

##GLOBAL CONSTANTS

#Critical density in units of
#h^2 M_sol Mpc^-3 (from Peacock book)
#Independent of cosmology, just definition of rho_crit
#and using H0 and G in adequate units
#This is adequate for the units used here
RHO_CRIT_UNITS = 2.7752E+11


#Define class and context to be able to 'silence' the output from
#some part of the code or specific funtion
#Taken from http://stackoverflow.com/a/2829036/2903411

class DummyFile(object):
    """
    Dummy file object to implement http://stackoverflow.com/a/2829036/2903411
    """
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    """
    New 'no output' context to implement http://stackoverflow.com/a/2829036/2903411
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout




def aux_pkint_hankel(y_kr, r, pkfunction):
    """
    Auxiliar function to use in pk2xir_hankel. This is the function in the integrand, expressed
    in terms of y=kr (this is the variable over which we perform the integral), and r.
    pkfunction is a function that returns the power spectrum P(k) (can be an interpolant over
    tabulated values)
    """

    return (y_kr**2)*pkfunction(y_kr/r)

    
    
#These values of hankelN, hankelh = (6000, 0.0005) seem to work very well over all the
#relevant scales using WMAP7_z0_lin, probably fine for all our purposes
def pk2xir_hankel(rvalues, pkfunction, hankelN=6000, hankelh=0.0005):
    """
    Function that makes the standard tranformation from P(k) to xi(r) performing a
    Hankel transform, using the functions in library 'hankel.py' by Steven Murray.

    We convert the standard transformation, given by

    \\xi(r) = \int_0^{+\infty} P(k) \frac{\sin(kr)}{kr} \frac{4 \pi k^2}{(2\pi)^3} dk ,

    by making the change of variables y = kr, which results in

    \\xi(r) = \frac{4\pi}{(2 \pi r)^3} \int_0^{+\infty} [y^2 P(y/r)] j_0(y) dy ,

    where j_0(y) = sin(y)/y
    """

    Nr = len(rvalues)

    xi_out = np.empty(Nr,float)

    #Define the needed Hankel Transform instance, avoid printing
    #lots of stuff coming from the definition
    with nostdout():
        sph_hankel = hankel.SphericalHankelTransform(nu=0, N=hankelN, h=hankelh)
    
    for i,r in enumerate(rvalues):
        func_int = lambda x: aux_pkint_hankel(y_kr=x, r=r, pkfunction=pkfunction)
        norm = 1./(2.*pow(np.pi, 2)*pow(r,3.))
        int_result = sph_hankel.transform(func_int)[0]
        xi_out[i] = norm*int_result

    return xi_out

    
class PowerSpectrum:
    """Simple class to contain a power spectrum sampled at a set of
       given values of the wavenumber k
    """
    
    def __init__(self, kvals, pkvals):
        """
        """

        if(kvals.ndim != 1 or pkvals.ndim !=1):
            raise TypeError("The k and pk values passed to the PowerSpectrum class should be 1-dimensional arrays!")

        self.N = len(kvals)

        if(self.N != len(pkvals)):
            raise ValueError("The k and pk arrays passed to the PowerSpectrum class do not have same length!")

        if((kvals<0).any()):
            raise ValueError("The k values passed to the PowerSpectrum class must be positive!")

        if((pkvals<0).any()):
            raise ValueError("The Pk values passed to the PowerSpectrum class must be positive!")

        sortind = kvals.argsort()

        if((sortind != range(self.N)).any()):
            self.k = kvals[sortind]
            self.pk = pkvals[sortind]
            raise UserWarning("k-values passed to PowerSpectrum class were not correctly ordered!")
        else:
            self.k = kvals
            self.pk = pkvals



    def pkinterp(self, kval):
        """ Function that interpolates linearly the power spectrum using the given values.
        """

        return np.interp(x=kval, xp=self.k, fp=self.pk)



    def xir(self, rvals, hankelN=6000, hankelh=0.0005):
        """
        Function that performs a Hankel transform to obtain the 2-point correlation function
        corresponding to the power spectrum.
        """

        xivals = pk2xir_hankel(rvalues=rvals, pkfunction=self.pkinterp,
                               hankelN=hankelN, hankelh=hankelh)

        return xivals

    
        
        
