

"""
   hods_hodmodel.py --- HOD model-related function/classes for the simple HOD model

   Author: P. Arnalte-Mur (ICC-Durham)
   Creation date: 19/02/2014
   Last modified: --

   This module will contain the classes and functions related to the HOD models
   (i.e. number of central/satellite galaxies per halo, etc.)

"""


import numpy as np
import astropy.cosmology as ac
import scipy.special as spc


class HODModel():
    """Class that defines an HOD model, i.e. mean number of central and satellite
       galaxies per halo, as function of halo mass
    """

    def __init__(self, hod_type=1, mass_min=1e11, mass_1=1e12, alpha=1.0, siglogM=1.0, mass_0=1e11):
        """Parameters defining the HOD model:

           hod_type: defines the functional form of the model:
                     hod_type=1 --> Kravtsov (2004) model
                     hod_type=2 --> Zheng (2005) model
           mass_min: minimum mass for a halo to contain a galaxy
           mass_1: mass of the haloes that contain, on average, one satellite galaxy
           alpha: slope of the power-law relation
           siglogM: width of the transition from 0 to 1 centrals. Only used if hod_type==2
           mass_0: minimum mass for a halo to contain a satellite galaxy. Only used if hod_type==2
        """

        if(hod_type not in [1,2]):
            raise ValueError("Allowed hod_type values are 1 (Kravtsov) and 2 (Zheng)")

        self.hod_type = hod_type
        self.mass_min = mass_min
        self.mass_1 = mass_1
        self.alpha = alpha

        if(hod_type == 2):
            self.siglogM = siglogM
            self.mass_0 = mass_0

    def n_centrals(self, mass=1e12):
        """Returns mean number of central galaxies in a halo of mass 'mass',
           according to this HOD model
        """

        if(self.hod_type == 1):
            if(mass > self.mass_min):
                nc = 1.0
            else:
                nc = 0.0

        elif(self.hod_type == 2):

            term = (np.log10(mass) - np.log10(self.mass_min))/self.siglogM
            nc = 0.5*(1. + spc.erf(term))

        return nc

    def n_satellites(self, mass=1e12):
        """Returns mean number of satellite galaxies in a halo of mass 'mass',
           according to this HOD model
        """

        if(self.hod_type == 1):
            ns = self.n_centrals(mass)*pow(mass/self.mass_1, self.alpha)

        if(self.hod_type == 2):
            if(mass > self.mass_0):
                ns = pow((mass - self.mass_0)/self.mass_1, self.alpha)
            else:
                ns = 0.0

        return ns

    def n_total(self, mass=1e12):
        """Returns total mean number of galaxies in a halo of mass 'mass',
           according to this HOD model
        """

        nt = self.n_centrals(mass) + self.n_satellites(mass)
        return nt

        
