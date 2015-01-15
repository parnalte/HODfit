

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
from scipy import integrate
import hods_halomodel as hm


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

           Adapted to work correctly when 'mass' is an array.
        """

        #Convert input to array if it is not, and check it is only 1D!
        mass = np.atleast_1d(mass)
        assert mass.ndim == 1
        
        if(self.hod_type == 1):

            #1 if m>mass_min, 0 if m<=mass_min
            nc = np.array((mass > self.mass_min), float)

        elif(self.hod_type == 2):

            term = (np.log10(mass) - np.log10(self.mass_min))/self.siglogM
            nc = 0.5*(1. + spc.erf(term))

        return nc

    def n_satellites(self, mass=1e12):
        """Returns mean number of satellite galaxies in a halo of mass 'mass',
           according to this HOD model.

           Adapted to work correctly when 'mass' is an array.
        """
        
        #Convert input to array if it is not, and check it is only 1D!
        mass = np.atleast_1d(mass)
        assert mass.ndim == 1

        if(self.hod_type == 1):
            ns = self.n_centrals(mass)*pow(mass/self.mass_1, self.alpha)

        if(self.hod_type == 2):
            ns = np.where(mass > self.mass_0, pow((mass - self.mass_0)/self.mass_1, self.alpha), 0.0)

        return ns

    def n_total(self, mass=1e12):
        """Returns total mean number of galaxies in a halo of mass 'mass',
           according to this HOD model
        """

        nt = self.n_centrals(mass) + self.n_satellites(mass)
        return nt

        
####################
##DERIVED QUANTITIES
###################

#Perform the integrations following that done for
#HaloModelMW02.integral_quantities()

        
def dens_galaxies(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the mean galaxy number density according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (14) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies': not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)

    return integrate.simps(y=(nt_gals*nd_diff_array), x=mass_array)
    
    
def bias_gal_mean(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the mean galaxy bias according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (13) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'bias_gal_mean': not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)
    bias_haloes_array = halo_instance.bias_fmass(mass=mass_array)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min, logM_max, logM_step)

    return integrate.simps(y=(bias_haloes_array*nt_gals*nd_diff_array), x=mass_array)/dens_gal_tot
    

def mean_halo_mass_hod(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the HOD-averaged mean halo mass according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (15) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

        if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'bias_gal_mean': not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min, logM_max, logM_step)

    return integrate.simps(y=(mass_array*nt_gals*nd_diff_array), x=mass_array)/dens_gal_tot

    
        
def dens_galaxies_old(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the mean galaxy number density according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (14) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)

    nbins = nsteps - 1

    sum_ngal = 0.

    for i in range(nbins):

        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Nt_gals_bin = hod_instance.n_total(M_mean)

        sum_ngal = sum_ngal + (Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

    return sum_ngal




def bias_gal_mean_old(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the mean galaxy bias according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (13) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)

    nbins = nsteps - 1

    sum_bgal = 0.
    sum_ndens = 0.

    for i in range(nbins):

        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Nt_gals_bin = hod_instance.n_total(M_mean)
        bias_bin = halo_instance.bias_fmass(mass=M_mean)


        sum_bgal = sum_bgal + (bias_bin*Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))
        sum_ndens = sum_ndens + (Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

    mean_bias_gals = sum_bgal/sum_ndens

    return mean_bias_gals
    



def mean_halo_mass_hod_old(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the HOD-averaged mean halo mass according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (15) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)

    nbins = nsteps - 1

    sum_mass = 0.
    sum_ndens = 0.

    for i in range(nbins):

        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Nt_gals_bin = hod_instance.n_total(M_mean)

        sum_mass = sum_mass + (M_mean*Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))
        sum_ndens = sum_ndens + (Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

    mean_mass = sum_mass/sum_ndens

    return mean_mass
    
    
def fraction_centrals(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the fraction of central galaxies per halo according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (16) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    #Check the mass array makes sense
    assert logM_min > 0 
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)

    nbins = nsteps - 1

    sum_cent = 0.
    sum_ndens = 0.

    for i in range(nbins):

        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Nt_gals_bin = hod_instance.n_total(M_mean)
        Nc_gals_bin = hod_instance.n_centrals(M_mean)

        sum_cent = sum_cent + (Nc_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))
        sum_ndens = sum_ndens + (Nt_gals_bin*halo_instance.ndens_differential(mass=M_mean)*(nu_2 - nu_1))

    fract_cent = sum_cent/sum_ndens

    return fract_cent



    
def fraction_satellites(hod_instance=None, halo_instance=None, logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):
    """Computes the fraction of satellites galaxies per halo according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (17) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    f_cent = fraction_centrals(hod_instance=hod_instance, halo_instance=halo_instance, logM_min=logM_min,
                               logM_max=logM_max, logM_step=logM_step)

    return 1.0 - f_cent


    
    
