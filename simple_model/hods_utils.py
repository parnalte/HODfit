"""
   hods_utils.py --- General utilities for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: 19/02/2014

"""


import numpy as np
import astropy.cosmology as ac

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
        
        
