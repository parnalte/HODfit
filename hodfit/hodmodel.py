
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
import astropy.cosmology as ac

import densprofile


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

def dens_galaxies(hod_instance=None, halo_instance=None, logM_min=10.0,
                  logM_max=16.0, logM_step=0.05):
    """Computes the mean galaxy number density according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (14) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    # Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'dens_galaxies': "
                          "not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)

    return integrate.simps(y=(nt_gals*nd_diff_array), x=mass_array,
                           even='first')


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


def bias_gal_mean(hod_instance=None, halo_instance=None, logM_min=10.0,
                  logM_max=16.0, logM_step=0.05):
    """Computes the mean galaxy bias according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (13) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    # Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'bias_gal_mean': "
                          "not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)
    bias_haloes_array = halo_instance.bias_fmass(mass=mass_array)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                 logM_max, logM_step)

    return integrate.simps(y=(bias_haloes_array*nt_gals*nd_diff_array),
                           x=mass_array, even='first')/dens_gal_tot


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


def mean_halo_mass_hod(hod_instance=None, halo_instance=None, logM_min=10.0,
                       logM_max=16.0, logM_step=0.05):
    """Computes the HOD-averaged mean halo mass according to the combination
       of a halo distribution model and an HOD model.
       Following eq. (15) in C2012.

       hod_instance: an instance of the HODModel class
       halo_instance: an instance of the hm.HaloModelMW02 class
    """

    # Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'mean_halo_mass_hod': "
                          "not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                 logM_max, logM_step)

    return integrate.simps(y=(mass_array*nt_gals*nd_diff_array),
                           x=mass_array, even='first')/dens_gal_tot


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


def fraction_centrals(hod_instance=None, halo_instance=None, logM_min=10.0,
                      logM_max=16.0, logM_step=0.05):
    """
    Computes the fraction of central galaxies per halo according to the
    combination of a halo distribution model and an HOD model.
    Following eq. (16) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class
    """

    # Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'fraction_centrals': "
                          "not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nc_gals = hod_instance.n_centrals(mass=mass_array)

    dens_gal_tot = dens_galaxies(hod_instance, halo_instance, logM_min,
                                 logM_max, logM_step)

    return integrate.simps(y=(nc_gals*nd_diff_array),
                           x=mass_array, even='first')/dens_gal_tot


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


def fraction_satellites(hod_instance=None, halo_instance=None, logM_min=10.0,
                        logM_max=16.0, logM_step=0.05):
    """
    Computes the fraction of satellites galaxies per halo according to the
    combination of a halo distribution model and an HOD model.
    Following eq. (17) in C2012.

    hod_instance: an instance of the HODModel class
    halo_instance: an instance of the hm.HaloModelMW02 class
    """

    f_cent = fraction_centrals(hod_instance=hod_instance,
                               halo_instance=halo_instance, logM_min=logM_min,
                               logM_max=logM_max, logM_step=logM_step)

    return 1.0 - f_cent


def fraction_satellites_array(hod_instance=None, halo_instance=None):
    """
    Computes the fraction of satellites galaxies per halo, in the same way as
    the previous function, but using the predefined mass-dependent quantities
    in the hod and halomodel objects.
    """

    f_cent = fraction_centrals_array(hod_instance, halo_instance)

    return 1.0 - f_cent


# Functions needed to compute M_lim following the method of
# Tinker et al. (2005)
# Calculation of Mlim using Tinker et al. (2005) method
def probability_nonoverlap(radius, mass_1, mass_2, redshift=0, cosmo=ac.WMAP7):
    """
    Computes the probability that two ellipsoidal haloes of masses mass_1
    and mass_2 do not overlap if they are separated by a given radius.

    Corresponds to equation (A.23) in Coupon et al. (2012).

    Assumes mass_1 and mass_2 are 1-D arrays each (with lengths N1, N2),
    and radius is also a 1-D array with length Nr.
    Returns a NrxN1xN2 3D array with the result of the function for all
    the values of the three input parameters.
    """

    radius = np.atleast_1d(radius)
    mass_1 = np.atleast_1d(mass_1)
    mass_2 = np.atleast_1d(mass_2)

    assert radius.ndim == 1
    assert mass_1.ndim == 1
    assert mass_2.ndim == 1

    rvir_1 = densprofile.rvir_from_mass(mass_1, redshift, cosmo)
    rvir_2 = densprofile.rvir_from_mass(mass_2, redshift, cosmo)

    # rvirmesh_1, rvirmesh_2 = np.meshgrid(rvir_1, rvir_2, indexing='ij')

    # xvar, yvar are already N1xN2 arrays

    # Use notation to create N1xN2 array from
    # http://stackoverflow.com/a/20677444
    # And combine with an 'generalized outer' product with radius
    # to create a NrxN1xN2 array
    # xvar = radius/(rvir_1[:,None] + rvir_2)
    xvar = np.multiply.outer(radius, 1/(rvir_1[:, None] + rvir_2),
                             dtype=np.float32)

    # All derived arrays are now NrxN1xN2
    yvar = (xvar - 0.8)/0.29

    # result = 3*pow(yvar, 2) - 2*pow(yvar, 3)
    result = yvar*yvar*(3 - 2*yvar)
    result[yvar < 0] = 0
    result[yvar > 1] = 1

    return result


def galdens_haloexclusion(radius, redshift=0, cosmo=ac.WMAP7,
                          hod_instance=None, halo_instance=None):
    """
    Computes the galaxy number density taking into account the halo
    exclusion for ellipsoidal haloes. This will be used to determine the
    corresponding M_lim for 2-halo term integrations.

    Corresponds to equation (A.22) of Coupon et al. (2012).

    Will follow in part what is done in dens_galaxies_arrays.

    Works correctly when 'radius' is a 1D array (returns array of the same
    length).
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # Same for the radius array
    radius = np.atleast_1d(radius)
    assert radius.ndim == 1
    Nr = len(radius)

    dens_ntot_1d = hod_instance.n_tot_array*halo_instance.ndens_diff_m_array

    # dens_ntot_1, dens_ntot_2 = np.meshgrid(dens_ntot_1d, dens_ntot_1d)

    prob_over_term = probability_nonoverlap(radius=radius,
                                            mass_1=hod_instance.mass_array,
                                            mass_2=hod_instance.mass_array,
                                            redshift=redshift, cosmo=cosmo)

    ndens_2 = np.empty(Nr)
    for i, r in enumerate(radius):
        integrand_2d = np.outer(dens_ntot_1d, dens_ntot_1d)*prob_over_term[i]

        # Do the 2D integral by using Simpson's rule twice, as shown in
        # http://stackoverflow.com/a/20677444
        ndens_2[i] = integrate.simps(y=integrate.simps(
                                        y=integrand_2d,
                                        x=hod_instance.mass_array),
                                     x=hod_instance.mass_array)

    return np.sqrt(ndens_2)


def dens_galaxies_varmasslim(hod_instance=None, halo_instance=None):
    """
    Computes the mean galaxy number density as function of the variable
    mass limit, using the predefined mass-dependent quantities in the
    hod and halomodel objects.

    We take the possible values of the upper mass limit to be the same as the
    mass array, so the output will be an array of length Nm.

    We use the cumulative trapezoidal rule to make things easier and faster.

    This will be used for the halo-exclusion procedure (see eq. A.21 in
    Coupon et al., 2012)
    """

    # First, need to check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    # integrand = hod_instance.n_tot_array*halo_instance.ndens_diff_m_array

    # dens_gals_masslim = integrate.cumtrapz(y=integrand,
    #                                       x=hod_instance.mass_array,
    #                                       initial=0)

    dens_gals_masslim = np.empty(hod_instance.Nm, float)

    for i, mlim in enumerate(hod_instance.mass_array):
        dens_gals_masslim[i] = dens_galaxies_arrays(hod_instance=hod_instance,
                                                   halo_instance=halo_instance,
                                                   mass_limit=mlim)

    return dens_gals_masslim
