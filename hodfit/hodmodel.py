
"""
hods_hodmodel.py --- HOD model-related function/classes for the simple
                     HOD model

Author: P. Arnalte-Mur (ICC-Durham)
Creation date: 19/02/2014
Last modified: --

This module will contain the classes and functions related to the HOD models
(i.e. number of central/satellite galaxies per halo, etc.)
"""


import numpy as np
import scipy.special as spc
from scipy import integrate


class HODModel():
    """
    Class that defines an HOD model, i.e. mean number of central and satellite
    galaxies per halo, as function of halo mass
    """

    def __init__(self, hod_type=1, mass_min=1e11, mass_1=1e12, alpha=1.0,
                 siglogM=1.0, mass_0=1e11):
        """
        Parameters defining the HOD model:

        hod_type: defines the functional form of the model:
            hod_type=1 --> Kravtsov (2004) model
            hod_type=2 --> Zheng (2005) model
        mass_min: minimum mass for a halo to contain a galaxy
        mass_1: mass of the haloes that contain, on average, one satellite
            galaxy
        alpha: slope of the power-law relation
        siglogM: width of the transition from 0 to 1 centrals.
            Only used if hod_type==2
        mass_0: minimum mass for a halo to contain a satellite galaxy.
            Only used if hod_type==2
        """

        if(hod_type not in [1, 2]):
            raise ValueError("Allowed hod_type values are "
                             "1 (Kravtsov) and 2 (Zheng)")

        self.hod_type = hod_type
        self.mass_min = mass_min
        self.mass_1 = mass_1
        self.alpha = alpha

        if(hod_type == 2):
            self.siglogM = siglogM
            self.mass_0 = mass_0

        # Mass-dependent arrays we will eventually use
        # Will need to be initialized elsewhere
        self.mass_array = None
        self.Nm = 0
        self.n_cent_array = None
        self.n_sat_array = None
        self.n_tot_array = None

    def n_centrals(self, mass=1e12):
        """Returns mean number of central galaxies in a halo of mass 'mass',
           according to this HOD model

           Adapted to work correctly when 'mass' is an array.
        """

        # Convert input to array if it is not, and check it is only 1D!
        mass = np.atleast_1d(mass)
        assert mass.ndim == 1

        if(self.hod_type == 1):

            # 1 if m>mass_min, 0 if m<=mass_min
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

        # Convert input to array if it is not, and check it is only 1D!
        mass = np.atleast_1d(mass)
        assert mass.ndim == 1

        if(self.hod_type == 1):
            ns = self.n_centrals(mass)*pow(mass/self.mass_1, self.alpha)

        if(self.hod_type == 2):
            ns = np.where(mass > self.mass_0,
                          pow((mass - self.mass_0)/self.mass_1, self.alpha),
                          0.0)

        return ns

    def n_total(self, mass=1e12):
        """Returns total mean number of galaxies in a halo of mass 'mass',
           according to this HOD model
        """

        nt = self.n_centrals(mass) + self.n_satellites(mass)
        return nt

    def set_mass_arrays(self, logM_min=10., logM_max=16., logM_step=0.05):
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

        if self.mass_array[0] > self.mass_min:
            raise Exception("Not using all the mass range allowed by HOD!")

        self.n_cent_array = self.n_centrals(mass=self.mass_array)
        self.n_sat_array = self.n_satellites(mass=self.mass_array)
        self.n_tot_array = self.n_total(mass=self.mass_array)


####################
# DERIVED QUANTITIES
###################

# Perform the integrations following that done for
# HaloModelMW02.integral_quantities()

def dens_galaxies_arrays(hod_instance=None, halo_instance=None,
                         mass_limit=None):
    """
    Computes the mean galaxy number density, in the same way as the previous
    function, but using the predefined mass-dependent quantities in the hod
    and halomodel objects.

    Added the option of an additional upper mass limit, as this will be
    useful for the calculation of the 2-halo term including halo exclusion.
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # Implement the upper mass limit if needed
    if mass_limit is not None:
        mlim_selection = np.array(hod_instance.mass_array <= mass_limit, int)
    else:
        mlim_selection = np.ones(hod_instance.Nm, int)

    # If all is good, then do the calculation directly
    dens_gals = integrate.simps(
        y=(hod_instance.n_tot_array*halo_instance.ndens_diff_m_array *
           mlim_selection),
        x=hod_instance.mass_array, even='first')
    return dens_gals


def dens_galaxies(hod_instance=None, halo_instance=None, logM_min=10.0,
                  logM_max=16.0, reltol=1e-5):
    """
    Computes the mean galaxy number density according to the combination of a
    halo distribution model and an HOD model.
    Following eq. (14) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class

    Use Romberg integration (from Scipy), to the relative tolerance set
    by the parameter 'reltol'.
    We do a change of variable to integrate over x=log10(M).
    """

    assert logM_min > 0
    assert logM_max > logM_min
    assert reltol > 0

    if 10**(logM_min) > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies_romb': "
                          "not using all the mass range allowed by HOD!")

    def integrand(x):
        return np.log(10)*(10**x)*hod_instance.n_total(mass=10**x) * \
            halo_instance.ndens_diff_m(mass=10**x)

    integ_result = integrate.romberg(integrand, a=logM_min, b=logM_max,
                                     tol=0, rtol=reltol, vec_func=True)

    return integ_result[0]


def bias_gal_mean_array(hod_instance=None, halo_instance=None):
    """
    Computes the mean galaxy bias, in the same way as the previous
    function, but using the predefined mass-dependent quantities in the hod
    and halomodel objects.
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # If all is good, do the calculation directly
    bias_integ = \
        integrate.simps(y=(halo_instance.bias_array*hod_instance.n_tot_array *
                           halo_instance.ndens_diff_m_array),
                        x=hod_instance.mass_array, even='first')

    gal_dens = dens_galaxies_arrays(hod_instance, halo_instance)

    return bias_integ/gal_dens


def bias_gal_mean(hod_instance=None, halo_instance=None, logM_min=10.0,
                  logM_max=16.0, reltol=1e-5):
    """
    Computes the mean galaxy bias according to the combination
    of a halo distribution model and an HOD model.
    Following eq. (13) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class

    Use Romberg integration (from Scipy), to the relative tolerance set
    by the parameter 'reltol'.
    We do a change of variable to integrate over x=log10(M).
    """

    assert logM_min > 0
    assert logM_max > logM_min
    assert reltol > 0

    if 10**(logM_min) > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies_romb': "
                          "not using all the mass range allowed by HOD!")

    def integrand(x):
        return np.log(10)*(10**x)*hod_instance.n_total(mass=10**x) * \
            halo_instance.ndens_diff_m(mass=10**x) * \
            halo_instance.bias_fmass(mass=10**x)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                 logM_max, reltol)

    integ_result = integrate.romberg(integrand, a=logM_min, b=logM_max, tol=0,
                                     rtol=reltol, vec_func=True)

    return integ_result[0]/dens_gal_tot


def mean_halo_mass_hod_array(hod_instance=None, halo_instance=None):
    """
    Computes the HOD-averaged mean halo mass, in the same way as the previous
    function, but using the predefined mass-dependent quantities in the hod
    and halomodel objects.
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # If all is good, do the calculation directly
    mass_integ =\
        integrate.simps(y=(hod_instance.mass_array*hod_instance.n_tot_array *
                           halo_instance.ndens_diff_m_array),
                        x=hod_instance.mass_array, even='first')
    gal_dens = dens_galaxies_arrays(hod_instance, halo_instance)

    return mass_integ/gal_dens


def mean_halo_mass_hod(hod_instance=None, halo_instance=None,
                       logM_min=10.0, logM_max=16.0, reltol=1e-5):
    """
    Computes the HOD-averaged mean halo mass according to the combination
    of a halo distribution model and an HOD model.
    Following eq. (13) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class

    Use Romberg integration (from Scipy), to the relative tolerance set
    by the parameter 'reltol'.
    We do a change of variable to integrate over x=log10(M).
    """

    assert logM_min > 0
    assert logM_max > logM_min
    assert reltol > 0

    if 10**(logM_min) > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies_romb': "
                          "not using all the mass range allowed by HOD!")

    def integrand(x):
        return np.log(10)*(10**(2*x))*hod_instance.n_total(mass=10**x) * \
            halo_instance.ndens_diff_m(mass=10**x)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                 logM_max, reltol)

    integ_result = integrate.romberg(integrand, a=logM_min, b=logM_max, tol=0,
                                     rtol=reltol, vec_func=True)

    return integ_result[0]/dens_gal_tot


def fraction_centrals_array(hod_instance=None, halo_instance=None):
    """
    Computes the fraction of central galaxies per halo, in the same way as
    the previous function, but using the predefined mass-dependent quantities
    in the hod and halomodel objects.
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # If all is good, do the calculation directly
    centrals_dens = integrate.simps(
        y=(hod_instance.n_cent_array*halo_instance.ndens_diff_m_array),
        x=hod_instance.mass_array, even='first')
    gal_dens = dens_galaxies_arrays(hod_instance, halo_instance)

    return centrals_dens/gal_dens


def fraction_centrals(hod_instance=None, halo_instance=None,
                           logM_min=10.0, logM_max=16.0, reltol=1e-5):
    """
    Computes the fraction of central galaxies per halo according to the
    combination of a halo distribution model and an HOD model.
    Following eq. (13) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class

    Use Romberg integration (from Scipy), to the relative tolerance set
    by the parameter 'reltol'.
    We do a change of variable to integrate over x=log10(M).
    """

    assert logM_min > 0
    assert logM_max > logM_min
    assert reltol > 0

    if 10**(logM_min) > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies_romb': "
                          "not using all the mass range allowed by HOD!")

    def integrand(x):
        return np.log(10)*(10**x)*hod_instance.n_centrals(mass=10**x) * \
            halo_instance.ndens_diff_m(mass=10**x)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                      logM_max, reltol)

    integ_result = integrate.romberg(integrand, a=logM_min, b=logM_max, tol=0,
                                     rtol=reltol, vec_func=True)

    return integ_result[0]/dens_gal_tot


def fraction_satellites_array(hod_instance=None, halo_instance=None):
    """
    Computes the fraction of satellites galaxies per halo, in the same way as
    the previous function, but using the predefined mass-dependent quantities
    in the hod and halomodel objects.
    """

    f_cent = fraction_centrals_array(hod_instance, halo_instance)

    return 1.0 - f_cent


def fraction_satellites(hod_instance=None, halo_instance=None,
                        logM_min=10.0, logM_max=16.0, reltol=1e-5):
    """
    Computes the fraction of satellites per halo according to the
    combination of a halo distribution model and an HOD model.
    Following eq. (13) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class

    Uses Romberg integration.
    """

    f_cent = fraction_centrals(hod_instance=hod_instance,
                               halo_instance=halo_instance,
                               logM_min=logM_min, logM_max=logM_max,
                               reltol=reltol)
    return 1.0 - f_cent
