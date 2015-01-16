
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
from scipy import integrate
import hods_halomodel as halomodel
import hods_hodmodel as hodmodel
import hods_densprofile as densprofile
from hods_utils import PowerSpectrum


################################
# Integral quantities needed for different terms
################################

#Basically, follow the same procedure as done for
#halomodel.HaloModelMW02.integral_quantities() and
#hodmodel.dens_galaxies


def integral_centsatterm(rvalues, hod_instance=None, halo_instance=None,
                         logM_min = 10.0, logM_max = 16.0, logM_step=0.05,
                         redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    This function computes the integral needed to get the central-satellite
    term in the HOD clustering model, at a particular value of the scale 'r',
    or an array of r values.
    Parameters 'redshift, cosmo, powep_lin_0' are needed to define the
    NFW profile at each value of the mass.
    Following eq. (4) in 'model_definition.tex'

    Adapted to work efficiently for input 'r' arrays.
    The routine will return an array of the same length as 'rvalues'
    """

    #Convert input to array if it is not, and check it is only 1D!
    rvalues = np.atleast_1d(rvalues)
    assert rvalues.ndim == 1
    Nr = len(rvalues)
    
    #Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    Nm = len(mass_array)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'integral_centsatterm':
                          not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nc_gals = hod_instance.n_centrals(mass=mass_array)
    ns_gals = hod_instance.n_satellites(mass=mass_array)

    profile_instance = densprofile.HaloProfileNFW(mass=mass_array,
                                                  redshift=redshift,
                                                  cosmo=cosmo,
                                                  powesp_lin_0=powesp_lin_0)
    
    #Compute the profile at all the scales 'rvalue' for all our mass values
    #Output array will have shape (Nr, Nm), which
    #is the correct one to pass to integration routine
    #Include the needed normalisation
    
    dprofile_config = profile_instance.profile_config(r=rvalues)/mass_array

    return integrate.simps(y=(nd_diff_array*nc_gals*ns_gals*dprofile_config),
                           x=mass_array)
    
    
def integral_satsatterm(kvalue, hod_instance=None, halo_instance=None,
                        logM_min = 10.0, logM_max = 16.0, logM_step=0.05,
                        redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    This function computes the integral needed to get the satellite-satellite
    term in the HOD clustering model, at a particular value of the
    wavenumber 'k'.
    Parameters 'redshift, cosmo, powep_lin_0' are needed to define the
    NFW profile at each value of the mass.
    Following eq. (6) in 'model_definition.tex'
    """

    #Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)
    nbins = nsteps - 1
    
    sum_integral = 0.

    for i in range(nbins):
        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Ns_gals_bin = hod_instance.n_satellites(M_mean)

        profile_instance = densprofile.HaloProfileNFW(mass=M_mean,
                                                      redshift=redshift,
                                                      cosmo=cosmo,
                                                      powesp_lin_0=powesp_lin_0)

        dens_profile_bin = np.absolute(profile_instance.profile_fourier(k=kvalue))
        ndens_bin = halo_instance.ndens_differential(mass=M_mean)
        
        sum_integral = sum_integral + (ndens_bin*pow(Ns_gals_bin,2)*pow(dens_profile_bin, 2)*(nu_2 - nu_1))

    return sum_integral




def integral_2hterm(kvalue, hod_instance=None, halo_instance=None,
                    logM_min = 10.0, logM_max = 16.0, logM_step=0.05,
                    redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    This function computes the integral needed to get the 2-halo term
    in the HOD clustering model, at a particular value of the wavenumber 'k'.
    Parameters 'redshift, cosmo, powep_lin_0' are needed to define the
    NFW profile at each value of the mass.
    Following eq. (7) in 'model_definition.tex'
    """

    #Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)
    nsteps = len(mass_array)
    nbins = nsteps - 1
    
    sum_integral = 0.

    for i in range(nbins):
        M_mean = np.sqrt(mass_array[i]*mass_array[i+1]) #logarithmic mean
        nu_1 = halo_instance.nu_variable(mass=mass_array[i])
        nu_2 = halo_instance.nu_variable(mass=mass_array[i+1])

        Nt_gals_bin = hod_instance.n_total(M_mean)
        bias_bin = halo_instance.bias_fmass(mass=M_mean)

        profile_instance = densprofile.HaloProfileNFW(mass=M_mean,
                                                      redshift=redshift,
                                                      cosmo=cosmo,
                                                      powesp_lin_0=powesp_lin_0)

        dens_profile_bin = np.absolute(profile_instance.profile_fourier(k=kvalue))
        ndens_bin = halo_instance.ndens_differential(mass=M_mean)

        
        sum_integral = sum_integral + (ndens_bin*Nt_gals_bin*bias_bin*dens_profile_bin*(nu_2 - nu_1))

    return sum_integral




class HODClustering():
    """
    Class that contains a full model for the galaxy clustering for
    a particular model, defined by
    Cosmology+redshift+halo model+HOD model+NFW profile
    """

    def __init__(self, redshift=0, cosmo=ac.WMAP7, powesp_matter=None,
                 hod_instance=None, halo_instance=None, powesp_lin_0=None,
                 logM_min = 10.0, logM_max = 16.0, logM_step = 0.05):

        assert redshift >= 0
        assert powesp_matter is not None
        assert hod_instance is not None
        assert halo_instance is not None
        assert powesp_lin_0 is not None
        assert logM_min > 0
        assert logM_max > logM_min
        assert logM_step > 0

        
        self.redshift = redshift
        self.cosmo = cosmo
        self.powesp_matter = powesp_matter
        self.hod = hod_instance
        self.halomodel = halo_instance
        self.powesp_lin_0 = powesp_lin_0
        self.logM_min = logM_min
        self.logM_max = logM_max
        self.logM_step = logM_step

        self.pk_satsat = None
        self.pk_2h     = None


        self.gal_dens = hodmodel.dens_galaxies(hod_instance=self.hod,
                                               halo_instance=self.halomodel,
                                               logM_min=self.logM_min,
                                               logM_max=self.logM_max,
                                               logM_step=self.logM_step)



    def xi_centsat(self, rvalues):
        """
        Computes the xi for the central-satellite term at the scales
        given by 'rvalues'
        """

        Nr = len(rvalues)

        int_cs_r = integral_centsatterm(rvalues=rvalues,
                                        hod_instance=self.hod,
                                        halo_instance=self.halomodel,
                                        logM_min=self.logM_min,
                                        logM_max=self.logM_max,
                                        logM_step=self.logM_step,
                                        redshift=self.redshift,
                                        cosmo=self.cosmo,
                                        powesp_lin_0=self.powesp_lin_0)

        xir_cs = (2.*int_cs_r/pow(self.gal_dens, 2)) - 1.

        return xir_cs


    def get_pk_satsat(self, kvalues):
        """
        Computes the power spectrum for the satellite-satellite term
        at the scales given by 'kvalues'.
        Stores the result in self.pk_satsat as a PowerSpectrum instance
        """

        Nk = len(kvalues)

        int_ss_k = np.empty(Nk,float)

        for i,k in enumerate(kvalues):

            print "Computing P_satsat for k-value %d of %d" % (i, Nk)
            
            int_ss_k[i] = integral_satsatterm(kvalue=k, hod_instance=self.hod,
                                              halo_instance=self.halomodel,
                                              logM_min=self.logM_min,
                                              logM_max=self.logM_max,
                                              logM_step=self.logM_step,
                                              redshift=self.redshift,
                                              cosmo=self.cosmo,
                                              powesp_lin_0=self.powesp_lin_0)

        pkvals = int_ss_k/pow(self.gal_dens, 2.)

        self.pk_satsat = PowerSpectrum(kvals=kvalues, pkvals=pkvals)
            



        
    def xi_satsat(self, rvalues):
        """
        Computes the xi for satellite-satellite term at the scales
        given by 'rvalues'
        First, we check if we have already computed the corresponding P(k)
        and, if that is not the case, we compute it.
        """

        if self.pk_satsat is None:
            #Use same k-values as in matter power spectrum
            kvalues = self.powesp_matter.k
            self.get_pk_satsat(kvalues=kvalues)

        xir_ss = self.pk_satsat.xir(rvals=rvalues)

        return xir_ss

    
    def get_pk_2h(self):
        """
        Computes the power spectrum for the 2-halo term.
        We compute it at the same k-values in which P(k) for matter is given.
        Stores the result in self.pk_2h as a PowerSpectrum instance
        """

        kvalues = self.powesp_matter.k
        
        Nk = len(kvalues)

        int_2h_k = np.empty(Nk,float)

        for i,k in enumerate(kvalues):

            print "Computing P_2h for k-value %d of %d" % (i, Nk)

            int_2h_k[i] = integral_2hterm(kvalue=k, hod_instance=self.hod,
                                          halo_instance=self.halomodel,
                                          logM_min=self.logM_min,
                                          logM_max=self.logM_max,
                                          logM_step=self.logM_step,
                                          redshift=self.redshift,
                                          cosmo=self.cosmo,
                                          powesp_lin_0=self.powesp_lin_0)


        pkvals = self.powesp_matter.pk*int_2h_k*int_2h_k/pow(self.gal_dens, 2)

        self.pk_2h = PowerSpectrum(kvals=kvalues, pkvals=pkvals)
            
        
    def xi_2h(self, rvalues):
        """
        Computes the xi for the 2-halo term at the scales given by 'rvalues'
        First, we check if we have already computed the corresponding
        P(k) and, if that is not the case, we compute it.
        """

        if self.pk_2h is None:
            self.get_pk_2h()

        xir_2h = self.pk_2h.xir(rvals=rvalues)

        return xir_2h


    def xi_1h(self, rvalues):
        """
        Computes the 1-halo correlation function term from combining
        cent-sat and sat-sat terms
        """

        xir_1h = self.xi_centsat(rvalues) + self.xi_satsat(rvalues)

        return xir_1h

    def xi_total(self, rvalues):
        """
        Computes the total (1h + 2h) correlation function from the
        previous functions
        """

        xir_total = 1. + self.xi_1h(rvalues) + self.xi_2h(rvalues)

        return xir_total


    def xi_all(self, rvalues):
        """
        Computes all the relevant correlation functions at once
        (just combine previous functions together)
        """

        xi_cs = self.xi_centsat(rvalues)
        xi_ss = self.xi_satsat(rvalues)
        xi_2h = self.xi_2h(rvalues)

        xi_1h = xi_cs + xi_ss

        xi_tot = 1. + xi_1h + xi_2h

        return xi_tot, xi_2h, xi_1h, xi_cs, xi_ss

        

        
        

        
    