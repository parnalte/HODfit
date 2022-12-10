
"""
    hods_clustering.py --- Clustering-related function/classes
        for the simple HOD model

    Author: P. Arnalte-Mur (ICC-Durham)
    Creation date: 19/02/2014
    Last modified: --

    This module will contain the classes and functions related to
    the clustering predictions (i.e. 1- and 2-halo terms for the
    power spectrum/correlation function)

    Apart from other assumptions implied elsewhere, we assume a
    Poisson distribution for the number of galaxies in each halo.
"""

import numpy as np
import astropy.cosmology as ac
import hankel
from scipy import integrate
from scipy import interpolate

from . import halomodel
from . import hodmodel
from . import densprofile
from .utils import PowerSpectrum, xir2wp_pi, get_camb_pk


################################
# Integral quantities needed for different terms
################################

# Basically, follow the same procedure as done for
# halomodel.HaloModel.integral_quantities() and
# hodmodel.dens_galaxies

def integral_centsatterm_array(rvalues, hod_instance=None, halo_instance=None,
                               dprofile_config=None, use_mvir_limit=True,
                               redshift=0, cosmo=ac.WMAP7):
    """
    This function computes the integral needed to get the central-satellite
    term in the HOD clustering model, at a particular value of the scale 'r',
    or an array of r values.
    In this case we use the pre-computed *normalised* config-space density
    profiles, which has to be passed to the function (dprofile_config).
    Following eq. (4) in 'model_definition.tex'

    Adapted to work efficiently for input 'r' arrays.
    The routine will return an array of the same length as 'rvalues'.

    In this version of the function, we work with the mass-dependent
    pre-computed quantities in the hod and halo instances.
    """

    # Convert input to array if it is not, and check it is only 1D!
    rvalues = np.atleast_1d(rvalues)
    assert rvalues.ndim == 1
    Nr = len(rvalues)

    # Check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    assert dprofile_config.shape == (Nr, hod_instance.Nm)

    # TODO: we can potentially eliminate all this 'use_mvir_limit' part, as
    # now the config.-space density profile is truncated at Mvir (so this
    # change in integration limit should have no effect)
    #
    # Implement the virial mass lower limit in the integration, as in
    # eq. (A17) in C2012
    #
    # We will create a (Nr, Nm) array containing 0 or 1 depending on whether
    # the mass at this point is larger than mvir(r) for r at this point
    if use_mvir_limit:
        # First, compute an array containing the values of mvir for the
        # scales r considered
        mvir_values = densprofile.massvir_from_radius(radius=rvalues,
                                                      redshift=redshift,
                                                      cosmo=cosmo)

        # Now, create a (Nm, Nr) array containing the mass_array values
        mass_array_2d = np.tile(np.atleast_2d(hod_instance.mass_array).T, Nr)

        # And the corresponding (Nr, Nm) array containing the mvir_values
        mvir_array_2d = np.tile(np.atleast_2d(mvir_values).T, hod_instance.Nm)

        # And compute the selection 2d array (as int)
        select_mvir_2d = np.array(mass_array_2d.T > mvir_array_2d, int)

    else:
        # If we do not make this selection, just create the corresponding
        # array accepting all mass values
        select_mvir_2d = np.ones((Nr, hod_instance.Nm), int)

    # When doing the integration, take into account the mvir limit
    return integrate.simps(
        y=(halo_instance.ndens_diff_m_array *
           hod_instance.n_cent_array * hod_instance.n_sat_array *
           dprofile_config*select_mvir_2d),
        x=hod_instance.mass_array, even='first')


def integral_satsatterm_array(kvalues, hod_instance=None, halo_instance=None,
                              dprofile_fourier=None):
    """
    This function computes the integral needed to get the satellite-satellite
    term in the HOD clustering model, at a particular value of the
    wavenumber 'k', or an array of k values.
    We use the pro-computed Fourier-space density profiles, which have to
    be passed to the function (dprofile_fourier).
    Following eq. (6) in 'model_definition.tex'

    Adapted to work efficiently for input 'k'arrays.
    The routine will return an array of the same length as 'kvalues'

    In this version of the function, we work with the mass-dependent
    pre-computed quantities in the hod and halo instances.
    """

    # Check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    assert dprofile_fourier.shape == (len(kvalues), hod_instance.Nm)

    dprof_term = pow(np.absolute(dprofile_fourier), 2)

    return integrate.simps(
        y=(halo_instance.ndens_diff_m_array*(hod_instance.n_sat_array**2) *
           dprof_term),
        x=hod_instance.mass_array, even='first')


def mlim_nprime_zheng(rscales, redshift=0, cosmo=ac.WMAP7, hod_instance=None,
                      halo_instance=None, logM_min=10.0, logM_step=0.05):
    """
    Computes the upper mass integration limit and the modified galaxy
    density (nprime) needed to implement the Zheng (2004) model for the
    calculation of the 2-halo term. These are given, respectively, by
    eqs. (B4) and (B6) in Tinker et al. (2005).
    """

    # Want 'rscales' to be a 1D array
    rscales = np.atleast_1d(rscales)
    assert rscales.ndim == 1
    Nr = len(rscales)

    # First, compute the upper mass limit: virial mass for haloes of radius
    # Rad = r/2
    mass_lim = densprofile.massvir_from_radius(radius=rscales/2.,
                                               redshift=redshift, cosmo=cosmo)

    # And now, compute the corresponding n_prime
    n_prime_array = np.empty(Nr, float)

    for i, mlim in enumerate(mass_lim):
        n_prime_array[i] = \
            hodmodel.dens_galaxies_arrays(hod_instance=hod_instance,
                                          halo_instance=halo_instance,
                                          mass_limit=mlim)

    return mass_lim, n_prime_array


def mlim_nprime_tinker(rscales, redshift=0, cosmo=ac.WMAP7, hod_instance=None,
                       halo_instance=None, prob_nonover_array=None):
    """
    Computes the upper mass integration limit and the modified galaxy density
    (nprime) needed to implement the Tinker et al. (2005) model for the
    calculation of the 2-halo term, taking into account elliptical halo
    exclusion. This is described in eqs. (A.21-A.23) of Coupon et al. (2012),
    and we'll use several functions defined in hodmodel.py
    """

    # Want 'rscales' to be a 1D array
    rscales = np.atleast_1d(rscales)
    assert rscales.ndim == 1
    Nr = len(rscales)

    # Compute the array of galaxy number densities as function of the possible
    # mass limits
    densgal_masslim = \
        hodmodel.dens_galaxies_varmasslim(hod_instance=hod_instance,
                                          halo_instance=halo_instance)

    # Now, obtain the values of nprime as function of rval,
    # and match it to the appropriate mass_lim
    n_prime_out = \
        hodmodel.galdens_haloexclusion(radius=rscales,
                                       redshift=redshift,
                                       cosmo=cosmo,
                                       hod_instance=hod_instance,
                                       halo_instance=halo_instance,
                                       prob_nonover_array=prob_nonover_array)

    mlim_indices = np.searchsorted(densgal_masslim, n_prime_out)
    # For the cases when we get out of bounds, take the last value of the mass
    mlim_indices[mlim_indices == hod_instance.Nm] = hod_instance.Nm - 1
    mass_lim_out = hod_instance.mass_array[mlim_indices]

    # And now, compute the corresponding n_prime
    n_prime_array = np.empty(Nr, float)

    for i, mlim in enumerate(mass_lim_out):
        n_prime_array[i] = \
            hodmodel.dens_galaxies_arrays(hod_instance=hod_instance,
                                          halo_instance=halo_instance,
                                          mass_limit=mlim)
    return mass_lim_out, n_prime_array


def integral_2hterm_array(kvalues, hod_instance=None, halo_instance=None,
                          dprofile_fourier=None, mass_limit=None):
    """
    This function computes the integral needed to get the 2-halo term
    in the HOD clustering model, at a particular value of the wavenumber 'k',
    or an array of k values.
    We use the pro-computed Fourier-space density profiles, which have to
    be passed to the function (dprofile_fourier).
    Following eq. (7) in 'model_definition.tex'

    Adapted to work efficiently for input 'k'arrays.
    The routine will return an array of the same length as 'kvalues'

    In this version of the function, we work with the mass-dependent
    pre-computed quantities in the hod and halo instances.
    We add the option of fixing a more restrictive upper mass limit, which is
    needed to implement halo exclusion.
    """

    # Check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    assert dprofile_fourier.shape == (len(kvalues), hod_instance.Nm)

    dprof_term = np.absolute(dprofile_fourier)

    # Implement the upper mass limit if needed
    if mass_limit is not None:
        mlim_selection = np.array(hod_instance.mass_array <= mass_limit, int)
    else:
        mlim_selection = np.ones(hod_instance.Nm, int)

#    return integrate.simps(
#        y=(halo_instance.ndens_diff_m_array * hod_instance.n_tot_array *
#           halo_instance.bias_array * dprof_term * mlim_selection),
#        x=hod_instance.mass_array, even='first')
    return integrate.trapz(
        y=(halo_instance.ndens_diff_m_array * hod_instance.n_tot_array *
           halo_instance.bias_array * dprof_term * mlim_selection),
        x=hod_instance.mass_array)


def integral_2hterm_masslimarray(kvalues, hod_instance=None,
                                 halo_instance=None, redshift=0,
                                 cosmo=ac.WMAP7, powesp_lin_0=None,
                                 dprofile_fourier=None, mass_limit=None):
    """
    This function computes the integral needed to get the 2-halo term
    in the HOD clustering model, at a particular value of the wavenumber 'k',
    or an array of k values.
    We use the pro-computed Fourier-space density profiles, which have to
    be passed to the function (dprofile_fourier).
    Following eq. (7) in 'model_definition.tex'

    This function is adapted to get as input an array of k-values
    [of length Nk], *and* an array of mass_limits [of length Nr].
    The output will be a 2D array of shape [Nk, Nr].
    """

    # Check that the array quantities are properly set and
    # match each other
    assert hod_instance.Nm > 0
    assert hod_instance.Nm == halo_instance.Nm
    assert (hod_instance.mass_array == halo_instance.mass_array).all()

    assert dprofile_fourier.shape == (len(kvalues), hod_instance.Nm)

    # If mass_limit==None, set it to the maximum mass in the arrays
    if mass_limit is None:
        mass_limit = hod_instance.mass_array[-1]

    # Convert to array (even if single number)
    mass_limit = np.atleast_1d(mass_limit)

    # To avoid problems later (extrapolation), set the maximum mass limit
    # to the maximum mass in the mass array
    M_max = hod_instance.mass_array[-1]
    mass_limit = np.where(mass_limit < M_max, mass_limit, M_max)

    dprof_term = np.absolute(dprofile_fourier)

    # Now, first compute the cumulative integral for all values of the
    # mass array
    # Shape of 'cumul_integral' will be [Nk, Nm]
    cumul_integral = integrate.cumtrapz(
        y=(halo_instance.ndens_diff_m_array * hod_instance.n_tot_array *
           halo_instance.bias_array * dprof_term),
        x=hod_instance.mass_array, initial=0)

    # I use 2D (linear) interpolation to this object (maybe don't needed,
    # as the k-values I'm going to use are always the same, but this
    # makes it simpler)
    # Transposes needed because of the definition of functions...
    interp_function = interpolate.interp2d(x=kvalues,
                                           y=hod_instance.mass_array,
                                           z=cumul_integral.T, kind='linear')

    # This is the desired output, with the correct shape [Nk, Nr]
    integ_2hterm = interp_function(kvalues, mass_limit).T

    return integ_2hterm


class HODClustering(object):
    """
    Class that contains a full model for the galaxy clustering for
    a particular model, defined by
    Cosmology+redshift+halo model+HOD model+ (modified) NFW profile.

    When both parameters f_gal and gamma are equal to 1, we use a NFW
    halo profile. Otherwise, we use the modified NFW profile defined in
    Watson et al. (2010).

    The parameter 'scale_dep_bias' determines whether we use the
    scale dependence of halo bias proposed by Tinker et al. (2005) or not.
    [For the 'simple model', use scale_dep_bias=False]

    The parameter 'use_mvir_limit' determines whether we use the
    lower limit equal to Mvir(r) in the integration of the central-sat
    term, following eq. (A.17) in C2012.
    [For the 'simple model', use use_mvir_limit=False]

    The parameter 'halo_exclusion_model' defines which model to use to
    take into account halo exclusion in the computation of the 2-halo
    term:
      - halo_exclusion_model = 0: no halo exclusion used, this corresponds
          to the 'simple model'
      - halo_exclusion_model = 1: halo exclusion implemented following
          Zheng (2004), as described in eqs. (B4-B9) in Tinker et al. (2005)
      - halo_exclusion_model = 2 (default): will correspond
          to Tinker et al. (2005) proposed model, as used by Coupon-2012

    We associate a fixed array of r values to this object (which will be used
    for all xi(r) computations, so that, in the case of halo_exclusion_model=2
    we can pre-compute the values of P(r, M1, M2) needed for the calculation).

    We add seven parameters (fprof_grid_*, fprof_ft_hankel, fprof_Nk_interp,
    fprof_Nm_interp), to be passed to the profile object (in the ModNFW case),
    and that will control the details of the calculation of the Fourier-space
    halo profiles.
    If these parameters are None, the defaults defined in the appropriate
    functions will be used.
    Note that this fprof_ft_hankel can be different from the ft_hankel used
    for the transformations xi(r) <--> P(k) done directly in this class.
    """

    def __init__(self, redshift=0, cosmo=ac.WMAP7, powesp_matter=None,
                 hod_instance=None, halo_instance=None, powesp_lin_0=None,
                 f_gal=1.0, gamma=1.0,
                 logM_min=10.0, logM_max=16.0, logM_step=0.05,
                 scale_dep_bias=True, use_mvir_limit=True,
                 halo_exclusion_model=2, ft_hankel=None,
                 rvalues=np.logspace(-1, 2, 100),
                 fprof_grid_log_krvir=None, fprof_grid_log_conc=None,
                 fprof_grid_gamma=None, fprof_grid_profile=None,
                 fprof_grid_rhos=None,
                 fprof_ft_hankel=None,
                 fprof_Nk_interp=None, fprof_Nm_interp=None):

        assert redshift >= 0
        assert powesp_matter is not None
        assert hod_instance is not None
        assert halo_instance is not None
        assert powesp_lin_0 is not None
        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        assert rvalues.ndim == 1
        assert (rvalues >= 0).all()

        self.redshift = redshift
        self.cosmo = cosmo
        self.powesp_matter = powesp_matter
        self.hod = hod_instance
        self.halomodel = halo_instance
        self.powesp_lin_0 = powesp_lin_0
        self.logM_min = logM_min
        self.logM_max = logM_max
        self.logM_step = logM_step
        self.scale_dep_bias = scale_dep_bias
        self.use_mvir_limit = use_mvir_limit
        self.halo_exclusion_model = halo_exclusion_model
        self.ft_hankel = ft_hankel

        self.fprof_grid_log_krvir = fprof_grid_log_krvir
        self.fprof_grid_log_conc = fprof_grid_log_conc
        self.fprof_grid_gamma = fprof_grid_gamma
        self.fprof_grid_profile = fprof_grid_profile
        self.fprof_grid_rhos = fprof_grid_rhos
        self.fprof_ft_hankel = fprof_ft_hankel
        self.fprof_Nk_interp = fprof_Nk_interp
        self.fprof_Nm_interp = fprof_Nm_interp

        self.pk_satsat = None
        self.pk_2h = None

        # When creating the instance, compute the mass-dependent quantities
        # that we will need later
        self.hod.set_mass_arrays(logM_min=self.logM_min,
                                 logM_max=self.logM_max,
                                 logM_step=self.logM_step)
        self.halomodel.set_mass_arrays(logM_min=self.logM_min,
                                       logM_max=self.logM_max,
                                       logM_step=self.logM_step)

        # Compute galaxy density for this model
        self.gal_dens = \
            hodmodel.dens_galaxies_arrays(hod_instance=self.hod,
                                          halo_instance=self.halomodel)

        # Create the profile instance, and pre-compute the Fourier-space
        # profile. Will compute it at the k values given by powesp_matter
        # Use NFW or modified profile depending on the values of f_gal, gamma
        self.kvals = self.powesp_matter.k

        self.f_gal = f_gal
        self.gamma = gamma

        if (self.f_gal == 1) and (self.gamma == 1):
            self.modify_NFW = False
            self.densprofile = \
                densprofile.HaloProfileNFW(mass=self.hod.mass_array,
                                           redshift=self.redshift,
                                           cosmo=self.cosmo,
                                           powesp_lin_0=self.powesp_lin_0)
            self.dprofile_fourier = \
                self.densprofile.profile_fourier(k=self.kvals)
        else:
            self.modify_NFW = True
            self.densprofile = \
                densprofile.HaloProfileModNFW(mass=self.hod.mass_array,
                                              f_gal=self.f_gal,
                                              gamma=self.gamma,
                                              redshift=self.redshift,
                                              cosmo=self.cosmo,
                                              powesp_lin_0=self.powesp_lin_0,
                                              four_grid_log_krvir=self.fprof_grid_log_krvir,
                                              four_grid_log_conc=self.fprof_grid_log_conc,
                                              four_grid_gamma=self.fprof_grid_gamma,
                                              four_grid_profile=self.fprof_grid_profile,
                                              four_grid_rhos=self.fprof_grid_rhos,
                                              fourier_ft_hankel=self.fprof_ft_hankel,
                                              fourier_Nk_interp=self.fprof_Nk_interp,
                                              fourier_Nm_interp=self.fprof_Nm_interp)
            self.dprofile_fourier = \
                self.densprofile.mod_profile_fourier(k=self.kvals)

        # Stuff related to the scales array and pre-computation of the
        # probability of no overlapping array
        self.rvalues = np.sort(rvalues)

        if self.halo_exclusion_model == 2:
            self.get_p_no_overlap()
        else:
            self.p_no_overlap = None

        # Also, given the scales array, compute the corresponding normalised
        # configuration-space halo density profile(s)
        if self.modify_NFW:
            self.dprofile_config = \
                self.densprofile.mod_profile_config_norm(r=self.rvalues)
        else:
            self.dprofile_config = \
                self.densprofile.profile_config_norm(r=self.rvalues)

    def get_p_no_overlap(self):
        """
        Function to compute the probability of no overlap between haloes for
        the values of the HODClustering object
        """
        self.p_no_overlap = \
            hodmodel.probability_nonoverlap(radius=self.rvalues,
                                            mass_1=self.hod.mass_array,
                                            mass_2=self.hod.mass_array,
                                            redshift=self.redshift,
                                            cosmo=self.cosmo)
        return self.p_no_overlap

    def update_hod(self, hod_instance):
        """
        Function to update only the HOD values for the HODClustering object.
        Avoid the need to re-read other parameters when we only whant to
        change HOD (most comman case, e.g. for fitting).
        """

        assert hod_instance is not None

        # Update the HOD value, recompute mass-dependent quantities
        # and density accordingly
        self.hod = hod_instance
        self.hod.set_mass_arrays(logM_min=self.logM_min,
                                 logM_max=self.logM_max,
                                 logM_step=self.logM_step)
        self.gal_dens = \
            hodmodel.dens_galaxies_arrays(hod_instance=self.hod,
                                          halo_instance=self.halomodel)

        # We will need to re-compute all clustering terms, so reset them
        self.pk_satsat = None
        self.pk_2h = None

    def update_rvalues(self, rvalues):
        """
        Function to update only the scale (r) values for the HODClustering
        object, including re-computation of the p_no_overlap, if needed.
        """
        assert rvalues.ndim == 1
        assert (rvalues >= 0).all()

        self.rvalues = np.sort(rvalues)

        if self.halo_exclusion_model == 2:
            self.get_p_no_overlap()
        else:
            self.p_no_overlap = None

        if self.modify_NFW:
            self.dprofile_config = \
                self.densprofile.mod_profile_config_norm(r=self.rvalues)
        else:
            self.dprofile_config = \
                self.densprofile.profile_config_norm(r=self.rvalues)

    def update_profile_params(self, f_gal, gamma):
        """
        Function to update only the parameters defining the Modified NFW
        profile for the HODClustering object.
        Avoid the need to re-read other parameters when we only want to change
        the profile paraemeters (most common case, e.g. for fitting).
        """

        self.f_gal = f_gal
        self.gamma = gamma

        # Set the profile object and compute at once both the Fourier-space
        # and configuration space profiles
        if (self.f_gal == 1) and (self.gamma == 1):
            self.modify_NFW = False
            self.densprofile = \
                densprofile.HaloProfileNFW(mass=self.hod.mass_array,
                                           redshift=self.redshift,
                                           cosmo=self.cosmo,
                                           powesp_lin_0=self.powesp_lin_0)
            self.dprofile_fourier = \
                self.densprofile.profile_fourier(k=self.kvals)
            self.dprofile_config = \
                self.densprofile.profile_config_norm(r=self.rvalues)
        else:
            self.modify_NFW = True
            self.densprofile = \
                densprofile.HaloProfileModNFW(mass=self.hod.mass_array,
                                              f_gal=self.f_gal,
                                              gamma=self.gamma,
                                              redshift=self.redshift,
                                              cosmo=self.cosmo,
                                              powesp_lin_0=self.powesp_lin_0,
                                              four_grid_log_krvir=self.fprof_grid_log_krvir,
                                              four_grid_log_conc=self.fprof_grid_log_conc,
                                              four_grid_gamma=self.fprof_grid_gamma,
                                              four_grid_profile=self.fprof_grid_profile,
                                              four_grid_rhos=self.fprof_grid_rhos,
                                              fourier_ft_hankel=self.fprof_ft_hankel,
                                              fourier_Nk_interp=self.fprof_Nk_interp,
                                              fourier_Nm_interp=self.fprof_Nm_interp)

            self.dprofile_fourier = \
                self.densprofile.mod_profile_fourier(k=self.kvals)
            self.dprofile_config = \
                self.densprofile.mod_profile_config_norm(r=self.rvalues)

        # We will need to re-compute all clustering terms, so reset them
        self.pk_satsat = None
        self.pk_2h = None

    def xi_centsat(self):
        """
        Computes the xi for the central-satellite term at the scales
        given by 'self.rvalues'
        """

        int_cs_r = \
            integral_centsatterm_array(rvalues=self.rvalues,
                                       hod_instance=self.hod,
                                       halo_instance=self.halomodel,
                                       dprofile_config=self.dprofile_config,
                                       use_mvir_limit=self.use_mvir_limit,
                                       redshift=self.redshift,
                                       cosmo=self.cosmo)

        xir_cs = (2.*int_cs_r/pow(self.gal_dens, 2)) - 1.

        return xir_cs

    def get_pk_satsat(self, kvalues):
        """
        Computes the power spectrum for the satellite-satellite term
        at the scales given by 'kvalues'.
        Stores the result in self.pk_satsat as a PowerSpectrum instance
        """

        int_ss_k = \
            integral_satsatterm_array(kvalues=kvalues,
                                      hod_instance=self.hod,
                                      halo_instance=self.halomodel,
                                      dprofile_fourier=self.dprofile_fourier)

        pkvals = int_ss_k/pow(self.gal_dens, 2.)

        self.pk_satsat = PowerSpectrum(kvals=kvalues, pkvals=pkvals)

    def xi_satsat(self):
        """
        Computes the xi for satellite-satellite term at the scales
        given by 'self.rvalues'
        First, we check if we have already computed the corresponding P(k)
        and, if that is not the case, we compute it.
        """

        if self.pk_satsat is None:
            # Use same k-values as in matter power spectrum
            kvalues = self.powesp_matter.k
            self.get_pk_satsat(kvalues=kvalues)

        xir_ss = self.pk_satsat.xir(rvals=self.rvalues,
                                    ft_hankel=self.ft_hankel)

        return xir_ss

    def get_pk_2h(self):
        """
        Computes the power spectrum for the 2-halo term, assuming a halo bias
        which is constant with scale (as given by the halomodel instance).
        We compute it at the same k-values in which P(k) for matter is given.
        Stores the result in self.pk_2h as a PowerSpectrum instance
        """

        kvalues = self.powesp_matter.k

        int_2h_k = integral_2hterm_array(kvalues=kvalues,
                                         hod_instance=self.hod,
                                         halo_instance=self.halomodel,
                                         dprofile_fourier=self.dprofile_fourier)
        pkvals = self.powesp_matter.pk*pow(int_2h_k/self.gal_dens, 2)

        self.pk_2h = PowerSpectrum(kvals=kvalues, pkvals=pkvals)

    def get_pk_2h_scale(self, mass_lim, nprime):
        """
        Computes the 2-halo power-spectrum for a fixed scale r, taking
        into account halo exclusion. This is given by eq. (B5) in
        Tinker-2005.
        The scale is defined by the mass limit and modified galaxy density,
        which should have been calculated according to the needed halo
        exclusion model.
        """

        kvalues = self.powesp_matter.k

        # Do the 2D-integral, adding the mass limit as upper integration limit
        int_2h_k = integral_2hterm_array(kvalues=kvalues,
                                         hod_instance=self.hod,
                                         halo_instance=self.halomodel,
                                         mass_limit=mass_lim,
                                         dprofile_fourier=self.dprofile_fourier)

        # Now, compute P(k) taking into account the modified galaxy density
        pkvals = self.powesp_matter.pk*pow(int_2h_k/nprime, 2)

        return PowerSpectrum(kvals=kvalues, pkvals=pkvals)

    def get_xir_2h_scalesarr(self, rvalues, masslimvals, nprimevals):
        """
        Computes the correlation function xi(r) for the two-halo term
        given a set of input r-values and their corresponding maximum
        mass limits, and modified galaxy densities densities.
        The latter two should have obtained from the needed halo exclusion
        model
        """

        # First check that the lengths are correct
        assert len(rvalues) == len(masslimvals)
        assert len(rvalues) == len(nprimevals)

        kvalues = self.powesp_matter.k

        # Get the corresponding values of the integral term (2D: [Nk, Nr])
        int_2h_kr = \
            integral_2hterm_masslimarray(kvalues=kvalues,
                                         hod_instance=self.hod,
                                         halo_instance=self.halomodel,
                                         mass_limit=masslimvals,
                                         dprofile_fourier=self.dprofile_fourier)

        # And now, get the factor that should multiply the matter power
        # spectrum in each case.
        # when n'=0 (typycally, masslim < M_min), set this factor to 0
        factor_2h_kr = np.where(nprimevals > 0,
                                pow(int_2h_kr/nprimevals, 2), 0)

        # Now, need a loop over r values to transform the power spectrum
        # in each case
        xiprime = np.empty(len(rvalues), float)
        for i, r in enumerate(rvalues):
            this_r_pkvals = self.powesp_matter.pk*factor_2h_kr[:, i]
            this_r_powesp = PowerSpectrum(kvals=kvalues, pkvals=this_r_pkvals)
            xiprime[i] = this_r_powesp.xir(r, ft_hankel=self.ft_hankel)

        # Finally, re-escale the xiprime we obtained following eq. (B9)
        # of Tinker-2005
        xir_2h = (pow(nprimevals/self.gal_dens, 2)*(1. + xiprime)) - 1.

        return xir_2h

    def xi_2h(self):
        """
        Computes the xi for the 2-halo term at the scales given by
        'self.rvalues'.
        First, we check if we have already computed the corresponding
        P(k) and, if that is not the case, we compute it.

        Depending on the 'scale_dep_bias' parameter, we implement
        the modification for scale-dependent bias, described by
        eq. (A13) in C2002
        """

        # No halo exclusion considered, just use 'the' 2h power spectrum
        if self.halo_exclusion_model == 0:
            if self.pk_2h is None:
                self.get_pk_2h()

            xir_2h = self.pk_2h.xir(rvals=self.rvalues,
                                    ft_hankel=self.ft_hankel)

        # Zheng's halo exclusion model
        elif self.halo_exclusion_model == 1:

            # xir_2h = np.empty(len(rvalues), float)
            mlimvals, nprimevals = \
                mlim_nprime_zheng(rscales=self.rvalues, redshift=self.redshift,
                                  cosmo=self.cosmo,
                                  hod_instance=self.hod,
                                  halo_instance=self.halomodel,
                                  logM_min=self.logM_min,
                                  logM_step=self.logM_step)
            xir_2h = self.get_xir_2h_scalesarr(rvalues=self.rvalues,
                                               masslimvals=mlimvals,
                                               nprimevals=nprimevals)

        # Tinker's halo exclusion model
        elif self.halo_exclusion_model == 2:

            mlimvals, nprimevals = \
                mlim_nprime_tinker(rscales=self.rvalues,
                                   redshift=self.redshift,
                                   cosmo=self.cosmo,
                                   hod_instance=self.hod,
                                   halo_instance=self.halomodel,
                                   prob_nonover_array=self.p_no_overlap)
            xir_2h = self.get_xir_2h_scalesarr(rvalues=self.rvalues,
                                               masslimvals=mlimvals,
                                               nprimevals=nprimevals)

        else:
            raise Exception(
                "This halo exclusion model is not implemented (yet)!")

        if self.scale_dep_bias:
            xi_matter = self.powesp_matter.xir(rvals=self.rvalues,
                                               ft_hankel=self.ft_hankel)
            bias_correction = \
                pow(1. + (1.17 * xi_matter), 1.49) /\
                pow(1. + (0.69 * xi_matter), 2.09)
            xir_2h = bias_correction*xir_2h

        return xir_2h

    def xi_1h(self):
        """
        Computes the 1-halo correlation function term from combining
        cent-sat and sat-sat terms
        """

        xir_1h = self.xi_centsat() + self.xi_satsat()

        return xir_1h

    def xi_total(self):
        """
        Computes the total (1h + 2h) correlation function from the
        previous functions
        """

        xir_total = 1. + self.xi_1h() + self.xi_2h()

        return xir_total

    def xi_all(self):
        """
        Computes all the relevant correlation functions at once
        (just combine previous functions together)
        """

        xi_cs = self.xi_centsat()
        xi_ss = self.xi_satsat()
        xi_2h = self.xi_2h()

        xi_1h = xi_cs + xi_ss

        xi_tot = 1. + xi_1h + xi_2h

        return xi_tot, xi_2h, xi_1h, xi_cs, xi_ss


def hod_from_parameters(redshift=0, OmegaM0=0.27, OmegaL0=0.73,
                        OmegaB0=0.046, H0=70.0, use_camb=True,
                        init_power_amplitude=2.2e-9,
                        init_power_spect_index=0.96,
                        camb_halofit_version=None,
                        camb_kmax=200.0,
                        camb_k_per_logint=30,
                        camb_pk_matter_nonlinear=True,
                        powesp_matter_file=None,
                        powesp_linz0_file=None,
                        hod_type=1, hod_mass_min=1e11, hod_mass_1=1e12,
                        hod_alpha=1.0, hod_siglogM=0.5, hod_mass_0=1e11,
                        f_gal=1.0, gamma=1.0,
                        logM_min=8.0, logM_max=16.0, logM_step=0.005,
                        scale_dep_bias=True, use_mvir_limit=True,
                        halo_exclusion_model=2,
                        mass_function_model='Sheth2001',
                        bias_function_model='Tinker2005',
                        delta_halo_mass=200.0,
                        bias_scale_factor=1.0,
                        hankelN=6000, hankelh=0.0005, rmin=0.01, rmax=100.0,
                        nr=100, rlog=True,
                        fprof_grid_log_krvir=None, fprof_grid_log_conc=None,
                        fprof_grid_gamma=None, fprof_grid_profile=None,
                        fprof_grid_rhos=None,
                        fprof_hankelN=12000, fprof_hankelh=1e-6,
                        fprof_Nk_interp=100, fprof_Nm_interp=100):
    """
    Construct an HODClustering object defining all the needed parameters.
    """

    assert redshift >= 0

    if redshift > 10:
        raise UserWarning("You entered a quite high value of redshift (>10). \
I am not sure these models are valid there, proceed at your own risk.")

    # First, build the Cosmology object
    # As we work always using distances in Mpc/h, we should be
    # fine setting H0=100
    assert OmegaM0 >= 0
    if (OmegaM0 + OmegaL0) != 1:
        raise UserWarning("You are using a non-flat cosmology. \
Are you sure that is what you really want?")
    cosmo_object = ac.LambdaCDM(H0=100, Om0=OmegaM0, Ode0=OmegaL0)

    # Calculate or read in the needed PowerSpectrum objects
    if use_camb:
        pk_matter_object = get_camb_pk(redshift=redshift,
                                       OmegaM0=OmegaM0, OmegaL0=OmegaL0,
                                       OmegaB0=OmegaB0, H0=H0,
                                       Pinit_As=init_power_amplitude,
                                       Pinit_n=init_power_spect_index,
                                       nonlinear=camb_pk_matter_nonlinear,
                                       halofit_model=camb_halofit_version,
                                       kmax=camb_kmax,
                                       k_per_logint=camb_k_per_logint)

        pk_linz0_object = get_camb_pk(redshift=0,
                                       OmegaM0=OmegaM0, OmegaL0=OmegaL0,
                                       OmegaB0=OmegaB0, H0=H0,
                                       Pinit_As=init_power_amplitude,
                                       Pinit_n=init_power_spect_index,
                                       nonlinear=False,
                                       kmax=camb_kmax,
                                       k_per_logint=camb_k_per_logint)

    else:
        pk_matter_object = PowerSpectrum.fromfile(powesp_matter_file)
        pk_linz0_object = PowerSpectrum.fromfile(powesp_linz0_file)

    # Build the HOD object
    hod_object = hodmodel.HODModel(hod_type=hod_type, mass_min=hod_mass_min,
                                   mass_1=hod_mass_1, alpha=hod_alpha,
                                   siglogM=hod_siglogM, mass_0=hod_mass_0)

    # Build the halo model object.
    # we just pass the parameters identifying the required HMF and bias
    # function models
    halo_object = halomodel.HaloModel(cosmo=cosmo_object,
                                      powesp_lin_0=pk_linz0_object,
                                      redshift=redshift,
                                      mass_function_model=mass_function_model,
                                      bias_function_model=bias_function_model,
                                      Delta=delta_halo_mass,
                                      scale_factor=bias_scale_factor,
                                      )

    # Now, create the Hankel FourierTransform object needed for the conversions
    # P(k) --> xi(r)
    ft_hankel = hankel.SymmetricFourierTransform(ndim=3, N=hankelN, h=hankelh)

    # Define the array of r-values for the HODClustering object
    assert rmax > rmin
    assert rmin >= 0
    assert nr > 0

    if rlog:
        rvals_array = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    else:
        rvals_array = np.linspace(rmin, rmax, nr)

    # Create the Hankel FourierTransform object corresponding to the conversion
    # of the halo radial profiles from config. to Fourier space
    fprof_ft_hankel = hankel.SymmetricFourierTransform(ndim=3, N=fprof_hankelN,
                                                       h=fprof_hankelh)

    # And finally, define the clustering object
    model_clustering_object = \
        HODClustering(redshift=redshift, cosmo=cosmo_object,
                      powesp_matter=pk_matter_object, hod_instance=hod_object,
                      halo_instance=halo_object, powesp_lin_0=pk_linz0_object,
                      f_gal=f_gal, gamma=gamma,
                      logM_min=logM_min, logM_max=logM_max,
                      logM_step=logM_step, scale_dep_bias=scale_dep_bias,
                      use_mvir_limit=use_mvir_limit,
                      halo_exclusion_model=halo_exclusion_model,
                      ft_hankel=ft_hankel, rvalues=rvals_array,
                      fprof_grid_log_krvir=fprof_grid_log_krvir,
                      fprof_grid_log_conc=fprof_grid_log_conc,
                      fprof_grid_gamma=fprof_grid_gamma,
                      fprof_grid_profile=fprof_grid_profile,
                      fprof_grid_rhos=fprof_grid_rhos,
                      fprof_ft_hankel=fprof_ft_hankel,
                      fprof_Nk_interp=fprof_Nk_interp,
                      fprof_Nm_interp=fprof_Nm_interp)

    print("New HODClustering object created, "
          f"galaxy density = {model_clustering_object.gal_dens:.4} (h/Mpc)^3")

    return model_clustering_object


def get_wptotal(rpvals, clustering_object, partial_terms=False, nr=300,
                pimin=0.001, pimax=500, npi=300):
    """
    Compute the total wp(rp) function given a HODClustering object.
    If partial_terms=True, return also the 1-halo and 2-halo terms
    """

    # First, checks on rp and convert to 1D array
    rpvals = np.atleast_1d(rpvals)
    assert rpvals.ndim == 1

    # Define the array in r we will use to compute the model xi(r)
    rmin = rpvals.min()
    rmax = np.sqrt((pimax**2.) + (rpvals.max()**2.))
    rarray = clustering_object.rvalues

    # Check that the array we want matches that already used in the
    # clustering object, and if not update it
    if (not np.isclose(rarray.min(), rmin)) or \
       (not np.isclose(rarray.max(), rmax)) or (len(rarray) != nr):

        rarray_new = np.logspace(np.log10(rmin), np.log10(rmax), nr)
        clustering_object.update_rvalues(rarray_new)
        rarray = rarray_new

    logpimin = np.log10(pimin)
    logpimax = np.log10(pimax)

    if partial_terms:

        # Obtain the needed xi(r) functions
        xitot, xi2h, xi1h, xics, xiss = clustering_object.xi_all()

        # And convert to wp(rp) functions
        wptot = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xitot,
                          logpimin=logpimin, logpimax=logpimax, npi=npi)
        wp2h = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xi2h,
                         logpimin=logpimin, logpimax=logpimax, npi=npi)
        wp1h = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xi1h,
                         logpimin=logpimin, logpimax=logpimax, npi=npi)
        wpcs = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xics,
                         logpimin=logpimin, logpimax=logpimax, npi=npi)
        wpss = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xiss,
                         logpimin=logpimin, logpimax=logpimax, npi=npi)

        return wptot, wp2h, wp1h, wpcs, wpss

    else:
        xitot = clustering_object.xi_total()

        wptot = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xitot,
                          logpimin=logpimin, logpimax=logpimax, npi=npi)

        return wptot
