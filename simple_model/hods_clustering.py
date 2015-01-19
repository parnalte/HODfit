
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
from hods_utils import PowerSpectrum, xir2wp_pi


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
        raise UserWarning("In function 'integral_centsatterm': \
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
    
    
def integral_satsatterm(kvalues, hod_instance=None, halo_instance=None,
                        logM_min = 10.0, logM_max = 16.0, logM_step=0.05,
                        redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    This function computes the integral needed to get the satellite-satellite
    term in the HOD clustering model, at a particular value of the
    wavenumber 'k', or an array of k values.
    Parameters 'redshift, cosmo, powep_lin_0' are needed to define the
    NFW profile at each value of the mass.
    Following eq. (6) in 'model_definition.tex'

    Adapted to work efficiently for input 'k'arrays.
    The routine will return an array of the same length as 'kvalues'
    """

    #Convert input to array if it is not, and check it is only 1D!
    kvalues = np.atleast_1d(kvalues)
    assert kvalues.ndim == 1
    Nk = len(kvalues)
    
    #Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)

    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'integral_satsatterm': \
                          not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    ns_gals = hod_instance.n_satellites(mass=mass_array)

    profile_instance = densprofile.HaloProfileNFW(mass=mass_array,
                                                  redshift=redshift,
                                                  cosmo=cosmo,
                                                  powesp_lin_0=powesp_lin_0)


    #Compute the Fourier-space profile at all the scales 'kvalues'
    #for all our mass values
    #Output array will have shape (Nk, Nm), which is the correct one to
    #pass to integration routine
    dprofile_fourier = profile_instance.profile_fourier(k=kvalues)
    dprof_term = pow(np.absolute(dprofile_fourier), 2)
    
    return integrate.simps(y=(nd_diff_array*ns_gals*ns_gals*dprof_term),
                           x=mass_array)



def integral_2hterm(kvalues, hod_instance=None, halo_instance=None,
                    logM_min = 10.0, logM_max = 16.0, logM_step=0.05,
                    redshift=0, cosmo=ac.WMAP7, powesp_lin_0=None):
    """
    This function computes the integral needed to get the 2-halo term
    in the HOD clustering model, at a particular value of the wavenumber 'k',
    or an array of k values.
    Parameters 'redshift, cosmo, powep_lin_0' are needed to define the
    NFW profile at each value of the mass.
    Following eq. (7) in 'model_definition.tex'

    Adapted to work efficiently for input 'k'arrays.
    The routine will return an array of the same length as 'kvalues'
    """

    #Convert input to array if it is not, and check it is only 1D!
    kvalues = np.atleast_1d(kvalues)
    assert kvalues.ndim == 1
    Nk = len(kvalues)
    
    #Check the mass array makes sense
    assert logM_min > 0
    assert logM_max > logM_min
    assert logM_step > 0

    mass_array = 10**np.arange(logM_min, logM_max, logM_step)


    if mass_array[0] > hod_instance.mass_min:
        raise UserWarning("In function 'integral_2hterm': \
                          not using all the mass range allowed by HOD!")

    nd_diff_array = halo_instance.ndens_diff_m(mass=mass_array)
    nt_gals = hod_instance.n_total(mass=mass_array)
    bias_h_array = halo_instance.bias_fmass(mass=mass_array)

    profile_instance = densprofile.HaloProfileNFW(mass=mass_array,
                                                  redshift=redshift,
                                                  cosmo=cosmo,
                                                  powesp_lin_0=powesp_lin_0)

    
    #Compute the Fourier-space profile at all the scales 'kvalues'
    #for all our mass values
    #Output array will have shape (Nk, Nm), which is the correct one to
    #pass to integration routine
    dprofile_fourier = profile_instance.profile_fourier(k=kvalues)
    dprof_term = np.absolute(dprofile_fourier)

    return integrate.simps(y=(nd_diff_array*nt_gals*bias_h_array*dprof_term),
                           x=mass_array)


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


    def update_hod(self, hod_instance):
        """
        Function to update only the HOD values for the HODClustering object.
        Avoid the need to re-read other parameters when we only whant to change HOD (most comman case,
        e.g. for fitting).
        """

        assert hod_instance is not None

        #Update the HOD value, and density accordingly
        self.hod = hod_instance
        self.gal_dens = hodmodel.dens_galaxies(hod_instance=self.hod,
                                               halo_instance=self.halomodel,
                                               logM_min=self.logM_min,
                                               logM_max=self.logM_max,
                                               logM_step=self.logM_step)

        #We will need to re-compute all clustering terms, so reset them
        self.pk_satsat = None
        self.pk_2h     = None

        
        

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

        int_ss_k = integral_satsatterm(kvalues=kvalues, hod_instance=self.hod,
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

        int_2h_k = integral_2hterm(kvalues=kvalues, hod_instance=self.hod,
                                          halo_instance=self.halomodel,
                                          logM_min=self.logM_min,
                                          logM_max=self.logM_max,
                                          logM_step=self.logM_step,
                                          redshift=self.redshift,
                                          cosmo=self.cosmo,
                                          powesp_lin_0=self.powesp_lin_0)
        
        pkvals = self.powesp_matter.pk*pow(int_2h_k/self.gal_dens, 2)
        
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

        
def hod_from_parameters(redshift=0, OmegaM0=0.27, OmegaL0=0.73,
                        powesp_matter_file="WMAP7_z0_matterpower.dat",
                        powesp_linz0_file="WMAP7_linz0_matterpower.dat",
                        hod_type=1, hod_mass_min=1e11, hod_mass_1=1e12,
                        hod_alpha=1.0, hod_siglogM=0.5, hod_mass_0=1e11,
                        logM_min=8.0, logM_max=16.0, logM_step=0.005):
    """
    Construct an HODClustering object defining all the needed parameters.
    """

    assert redshift >= 0

    if redshift > 10:
        raise UserWarning("You entered a quite high value of redshift (>10). \
                          I am not sure these models are valid there, proceed at your \
                          own risk.")
    
    #First, build the Cosmology object
    #As we work always using distances in Mpc/h, we should be
    #fine setting H0=100
    assert OmegaM0 >= 0
    if (OmegaM0 + OmegaL0) != 1:
        raise UserWarning("You are using a non-flat cosmology. Are you sure that is \
                          what you really want?")
    cosmo_object = ac.LambdaCDM(H0=100, Om0=OmegaM0, Ode0=OmegaL0)

    #Build the needed PowerSpectrum objects
    try:
        km, pkm = np.loadtxt(powesp_matter_file, usecols=range(2), unpack=True)
        pk_matter_object = PowerSpectrum(kvals=km, pkvals=pkm)
    except:
        raise ValueError("Error reading matter power spectrum file %s" % powesp_matter_file)

    try:
        kl, pkl = np.loadtxt(powesp_linz0_file, usecols=range(2), unpack=True)
        pk_linz0_object = PowerSpectrum(kvals=kl, pkvals=pkl)
    except:
        raise ValueError("Error reading z=0 linear matter power spectrum file %s" % powesp_linz0_file)

    #Build the HOD object
    hod_object = hodmodel.HODModel(hod_type=hod_type, mass_min=hod_mass_min,
                                   mass_1=hod_mass_1, alpha=hod_alpha,
                                   siglogM=hod_siglogM, mass_0=hod_mass_0)


    #Build the halo model object
    halo_object = halomodel.HaloModelMW02(cosmo=cosmo_object,
                                          powesp_lin_0=pk_linz0_object,
                                          redshift=redshift)

    #And finally, define the clustering object
    model_clustering_object = \
        HODClustering(redshift=redshift, cosmo=cosmo_object,
                      powesp_matter=pk_matter_object, hod_instance=hod_object,
                      halo_instance=halo_object, powesp_lin_0=pk_linz0_object,
                      logM_min=logM_min, logM_max=logM_max, logM_step=logM_step)

    print "New HODClustering object created, \
galaxy density = %.4g (h/Mpc)^3 " % model_clustering_object.gal_dens

    return model_clustering_object
    
    
        
        
def get_wptotal(rpvals, clustering_object, partial_terms=False, nr=500, pimin=0.001, pimax=500, npi=500):
    """
    Compute the total wp(rp) function given a HODClustering object.
    If partial_terms=True, return also the 1-halo and 2-halo terms
    """

    #First, checks on rp and convert to 1D array
    rpvals = np.atleast_1d(rpvals)
    assert rpvals.ndim == 1
    Nrp = len(rpvals)

    
    #Define the array in r we will use to compute the model xi(r)
    rmin = rpvals.min()
    rmax = np.sqrt((pimax*2.) + (rpvals.max()**2.))
    rarray = np.logspace(np.log10(rmin), np.log10(rmax), nr)

    logpimin = np.log10(pimin)
    logpimax = np.log10(pimax)

    if partial_terms:

        #Obtain the needed xi(r) functions
        xitot, xi2h, xi1h, xics, xiss = \
                clustering_object.xi_all(rvalues=rarray)

        #And convert to wp(rp) functions
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
        xitot = clustering_object.xi_total(rvalues=rarray)

        wptot = xir2wp_pi(rpvals=rpvals, rvals=rarray, xivals=xitot,
                          logpimin=logpimin, logpimax=logpimax, npi=npi)

        return wptot

