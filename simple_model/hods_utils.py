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

##GLOBAL CONSTANTS

#Critical density in units of
#h^2 M_sol Mpc^-3 (from Peacock book)
#Independent of cosmology, just definition of rho_crit
#and using H0 and G in adequate units
#This is adequate for the units used here
RHO_CRIT_UNITS = 2.7752E+11



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


    def xir(self, rvals):
        """ Function that performs a Hankel transform to obtain the 2-point correlation function
            corresponding to the power spectrum.
            Using the Hankel library from CosmoPy.
        """

        hankel_instance = hankel.Hankel(dim=3)
        xivals = hankel_instance.transform(f=self.pkinterp, x=rvals, pk2xi=True)

        return xivals

        
        
