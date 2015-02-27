
"""
hods_fit.py --- Functions/classes related to fitting the HOD models to data

Author: P. Arnalte-Mur (ICC-Durham)
Creation date: 26/02/2015
Last modified: --

This module will contain the classes and functions related to the ways
to fit the HOD model to different data (basically, correlation functions,
xi(r) or wp(rp), and galaxy number density).
Fitting will be based on the scipy.optimize and emcee libraries.

Some of these functions are taken/adapted from the emcee tutorials.
"""

import numpy as np
from scipy import optimize
import emcee

import hods_hodmodel as hodmodel
import hods_clustering as clustering






def wp_hod(rp, hod_params, clustobj=None, hod_type=1, nr=100, pimin=0.001,
           pimax=400, npi=100):
    """
    Basic function to compute wp(rp) for likelihood computations.

    Will use the parameters of the model defined in 'clustobj', which is an
    instance of a HODClustering class, except for the parameters of the HOD,
    which we assume that will be changing between likelihood calls.
    Mass parameters in HOD are assumed to be given as log10(M_x).

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class)
    """

    #First, define the new HOD given the parameters
    if hod_type==1:
        log10Mmin, log10M1, alpha = hod_params
        new_hod = hodmodel.HODModel(hod_type=1, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha)
    elif hod_type==2:
        log10Mmin, log10M1, alpha, siglogM, log10M0 = hod_params
        new_hod = hodmodel.HODModel(hod_type=2, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha,
                                    siglogM=siglogM, mass_0=10**log10M0)
    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    #Now, update the hodclustering object
    clustobj.update_hod(new_hod)
    
    #And compute the wp values (use default values for the details)
    return clustering.get_wptotal(rpvals=rp, clustering_object=clustobj,
                                  nr=nr, pimin=pimin, pimax=pimax, npi=npi)

    
def chi2_fullmatrix(data_vals, inv_covmat, model_predictions):
    """
    Given a set of data points, its inverse covariance matrix and the
    corresponding set of model predictions, computes the standard chi^2
    statistic (using the full covariances)
    """
    
    y_diff = data_vals - model_predictions
    return np.dot(y_diff, np.dot(inv_covmat, y_diff))

    
def lnprior_flat(hod_params, param_lims, hod_type=1):
    """
    Returns the (un-normalised) log(P) for a flat prior on the HOD parameters.
    Mass parameters are assumed to be given as log10(M_x) (so the prior will
    be flat on the latter), alpha and sigma_logM are assumed to be given
    directly.

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class).    
    """

    if hod_type==1:
        logMmin, logM1, alpha = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max = param_lims
        
        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max:
            return 0.0
        else:
            return -np.inf

    elif hod_type==2:
        logMmin, logM1, alpha, siglogM, logM0 = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max, siglogM_min, siglogM_max, \
            logM0_min, logM0_max = param_lims

        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max and \
           siglogM_min < siglogM < siglogM_max and \
           logM0_min < logM0 < logM0_max:
            return 0.0
        else:
            return -np.inf

    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)


def lnlikelihood_fullmatrix(hod_params, rp, wp, wp_icov, clustobj=None,
                            hod_type=1, nr=100, pimin=0.001, pimax=400, npi=100):
    """
    Computes the (un-normalised) log-likelihood of the data wp(rp) given
    its inverse covariance matrix, and the given values of the HOD parameters.

    The rest of the parameters of the model are set in 'clustobj', which
    is an instance of the HODClustering class.

    We assume the errors on wp come from a multi-dimensional Gaussian, and
    use the full covariances.

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class)
    """

    wp_model = wp_hod(rp=rp, hod_params=hod_params, clustobj=clustobj,
                      hod_type=hod_type, nr=nr, pimin=pimin, pimax=pimax,
                      npi=npi)
    
    return -0.5*chi2_fullmatrix(data_vals=wp, inv_covmat=wp_icov,
                                model_predictions=wp_model)


    

def lnposterior(hod_params, rp, wp, wp_icov, param_lims, clustobj=None,
                hod_type=1, nr=100, pimin=0.001, pimax=400, npi=100):
    """
    Computes the (un-normalised) log(P) of the posterior PDF of the HOD
    parameters given the wp(rp) data.

    Assumptions made in this implementation:
    * Prior is flat on the parameters (in log(M_x) for mass-like parameters).
    * Compute the likelihood using the full covariance matrix of the data,
      assuming multi-dimensional Gaussian errors.
    * We allow the use of different HOD parameterisations using the 'hod_type'
      parameter (same options as in HODModel class).
    """
    
    lprior = lnprior_flat(hod_params, param_lims, hod_type)
    
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlikelihood_fullmatrix(hod_params, rp, wp, wp_icov,
                                            clustobj, hod_type, nr, pimin,
                                            pimax, npi)



def select_scales(rpmin, rpmax, rp, wp, wperr=None, wp_covmatrix=None):
    """
    Do the scale selection for arrays that depend on scale:
    rp, wp, wp errors, and wp covariance matrix (the last two are optional)
    """

    #Check everything makes sense
    assert rpmin >=0
    assert rpmax > rpmin

    Nr = len(rp)
    assert len(wp) == Nr

    if wperr is not None:
        assert len(wperr) == Nr

    if wp_covmatrix is not None:
        assert wp_covmatrix.shape == (Nr,Nr)

    #Define the selection
    scale_selection = (rpmin<rp)*(rp<rpmax)

    #And apply it to all the arrays
    rp = rp[scale_selection]
    wp = wp[scale_selection]

    if wperr is not None:
        wperr = wperr[scale_selection]

    if wp_covmatrix is not None:
        wp_covmatrix = wp_covmatrix[scale_selection][:,scale_selection]

    return rp, wp, wperr, wp_covmatrix
        
