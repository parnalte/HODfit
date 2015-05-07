
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

import os

import numpy as np
from scipy import optimize
import emcee

import hods_hodmodel as hodmodel
import hods_clustering as clustering


def get_hod_from_params(hod_params, hod_type=1):
    """
    Define a new HOD class instance from the parameters.
    (This should probably be moved to the hods_hodmodel.py library!)
    """

    if hod_type == 1:
        log10Mmin, log10M1, alpha = hod_params
        new_hod = hodmodel.HODModel(hod_type=1, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha)
    elif hod_type == 2:
        log10Mmin, log10M1, alpha, siglogM, log10M0 = hod_params
        new_hod = hodmodel.HODModel(hod_type=2, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha,
                                    siglogM=siglogM, mass_0=10**log10M0)
    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    return new_hod


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

    # First, define the new HOD given the parameters
    new_hod = get_hod_from_params(hod_params, hod_type)

    # Now, update the hodclustering object
    clustobj.update_hod(new_hod)

    # And compute the wp values (use default values for the details)
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

    if hod_type == 1:
        logMmin, logM1, alpha = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max = param_lims

        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max:
            return 0.0
        else:
            return -np.inf

    elif hod_type == 2:
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
                            hod_type=1, nr=100, pimin=0.001, pimax=400,
                            npi=100):
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

    lp = lnprior_flat(hod_params, param_lims, hod_type)

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

    # Check everything makes sense
    assert rpmin >= 0
    assert rpmax > rpmin

    Nr = len(rp)
    assert len(wp) == Nr

    if wperr is not None:
        assert len(wperr) == Nr

    if wp_covmatrix is not None:
        assert wp_covmatrix.shape == (Nr, Nr)

    # Define the selection
    scale_selection = (rpmin < rp)*(rp < rpmax)

    # And apply it to all the arrays
    rp = rp[scale_selection]
    wp = wp[scale_selection]

    if wperr is not None:
        wperr = wperr[scale_selection]

    if wp_covmatrix is not None:
        wp_covmatrix = wp_covmatrix[scale_selection][:, scale_selection]

    return rp, wp, wperr, wp_covmatrix


def find_best_fit(hod_params_start, rp, wp, wp_icov, param_lims,
                  return_model=False, minim_method='Powell', clustobj=None,
                  hod_type=1, nr=100, pimin=0.001, pimax=400, npi=100):
    """
    Function to obtain the best-fit values of the HOD parameters, obtaining
    the maximum of the posterior function (equivalent to the maximum
    likelihood result when using flat priors).

    The maximization is done using standard scipy.optimize functions.
    'minim_method' defines the actual method used (this will be passed to
    optimize.minimize)

    The output of the function is defined by return_model:
    * If return_model==False:
        Just return a tuple containing the HOD parameter values for the best
        fit
    * If return_model==True:
        Returns (hod_params_best, best_model), where best_model is a tuple
        containing some characteristics of the best-fit model:
        (wp_array, galaxy_density, mean_halo_mass, mean_galaxy_bias, frac_sat)
    """

    # First, define the function to minimize (= -log-posterior)
    neglogposterior = lambda *args: -lnposterior(*args)

    # Now, do the actual minimization calling the function
    maxpost_result = \
        optimize.minimize(fun=neglogposterior, x0=hod_params_start,
                          args=(rp, wp, wp_icov, param_lims, clustobj,
                                hod_type, nr, pimin, pimax, npi),
                          method=minim_method)

    # Actually print the results
    print "Results of the maximization of the log(Posterior):"
    print maxpost_result

    # Get the best parameters values for output
    hod_params_best = maxpost_result['x']

    # Now, if needed, get other results for this model
    if return_model:
        wp_best = wp_hod(rp, hod_params_best, clustobj, hod_type, nr,
                         pimin, pimax, npi)
        hod_best = get_hod_from_params(hod_params_best, hod_type)
        galdens_best = \
            hodmodel.dens_galaxies_arrays(hod_instance=hod_best,
                                          halo_instance=clustobj.halomodel)
        meanhalomass_best = \
            hodmodel.mean_halo_mass_hod_array(hod_instance=hod_best,
                                              halo_instance=clustobj.halomodel)
        meangalbias_best = \
            hodmodel.bias_gal_mean_array(hod_instance=hod_best,
                                         halo_instance=clustobj.halomodel)

        fracsat_best = \
            hodmodel.fraction_satellites_array(hod_instance=hod_best,
                                              halo_instance=clustobj.halomodel)

        return hod_params_best, (wp_best, galdens_best,
                                 meanhalomass_best, meangalbias_best,
                                 fracsat_best)

    else:
        return hod_params_best


def get_initial_walker_positions(n_dimensions=3, n_walkers=100, init_type=0,
                                 param_lims=[-1, 1, -1, 1, -1, 1],
                                 central_position=[0, 0, 0],
                                 ball_size=[0.1, 0.1, 0.1]):
    """
    Function to get the initial positions of walkers in parameter space.
    I implement (for now?) two different recipes, to be chosen by parameter
    'init_type':

    - init_type=0: distribute initial points uniformly inside parameter limits
                   (anything fancy, so basically a flat prior on the parameters
                   considered)
                   parameter limits given by 'param_lims'
    - init_type=1: distribute initial points in a Gaussian 'ball' (ellipsoid?)
                   around a central point in parameter space (typically,
                   close to the best-fit solution).
                   This is recommended in
                   http://dan.iel.fm/emcee/current/user/line/
                   central point given by 'central_position'
                   size of 'ball' in each dimension given by 'ball_size'
    """

    if init_type == 0:
        assert len(param_lims) == n_dimensions*2

        # First create random array of numbers between 0 and 1
        # with the desired shape
        positions = np.random.rand(n_walkers, n_dimensions)

        # And now, scale each of the dimensions to the desired interval
        for i in range(n_dimensions):

            # param_lims will follow this convention, as in other functions
            # here we just make it general, for any number of parameters
            d_min = param_lims[2*i]
            d_max = param_lims[2*i + 1]
            assert d_max > d_min
            positions[:, i] = (d_max - d_min)*positions[:, i] + d_min

    elif init_type == 1:
        assert len(central_position) == n_dimensions
        assert len(ball_size) == n_dimensions

        # Just use code from emcee example
        positions = [central_position +
                     ball_size*np.random.randn(n_dimensions)
                     for i in range(n_walkers)]
        positions = np.array(positions)

    else:
        raise ValueError("The initialisation of walkers with"
                         "init_type = %d has not yet been implemented!"
                         % init_type)

    return positions


def run_mcmc(rp, wp, wp_icov, param_lims, clustobj=None, hod_type=1,
             nr=100, pimin=0.001, pimax=400, npi=100,
             init_type=0, cent_pos=None, ball_size=None,
             n_walkers=100, n_steps_per_walker=100, n_threads=1,
             out_chain_file="chain.default"):
    """
    Function to run the basic MCMC from emcee given the data, and the
    parameters defining the model.

    TODO: description of input parameters and options

    TODO: implement (including auxiliar functions) the option of taking into
          account the number density of the sample in the fit

    TODO: implement option to return also derived quantities for each sample
          in the chain
    """

    # First, check if file already exists (and is not emtpy!)
    # We do not want to overwrite anything
    if os.path.exists(out_chain_file) and\
            (os.stat(out_chain_file).st_size > 0):

        raise RuntimeError("File " + out_chain_file + " already exists and is "
                           "not empty. I will not overwrite anything!")

    # Depending on HOD type considered, get number of dimensions,
    # and header for the output file
    if hod_type == 1:
        n_dimensions = 3
        header = "walker logMmin logM1 alpha\n"

    elif hod_type == 2:
        n_dimensions = 5
        header = "walker logMmin logM1 alpha siglogM logM0\n"
    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    # Write header to output file (so we make sure it exists later!)
    f = open(out_chain_file, 'w')
    f.write(header)
    f.close()

    # Define initial positions for walkers
    initial_positions = \
        get_initial_walker_positions(n_dimensions=n_dimensions,
                                     n_walkers=n_walkers, init_type=init_type,
                                     param_lims=param_lims,
                                     central_position=cent_pos,
                                     ball_size=ball_size)

    # Now, define the emcee sampler
    sampler = \
        emcee.EnsembleSampler(nwalkers=n_walkers, dim=n_dimensions,
                              lnpostfn=lnposterior, threads=n_threads,
                              args=(rp, wp, wp_icov, param_lims, clustobj,
                                    hod_type, nr, pimin, pimax, npi))

    # And iterate the sampler, writing each of the samples to the output
    # chain file
    for result in sampler.sample(initial_positions,
                                 iterations=n_steps_per_walker,
                                 storechain=False):
        position = result[0]
        with open(out_chain_file, "a") as f:
            for k in range(position.shape[0]):
                f.write("%d  %s\n" % (k, str(position[k])[1:-1]))

    # If we get this far, we are finished!
    print "MCMC samples in run written to file ", out_chain_file

    return 0


def read_chain_file(inchain_file="chain.default"):
    """
    Reads in data from a 'chains file' with the same format as written out
    by function'run_mcmc'.
    
    In particular, note we read in the parameter names from the header of the
    file, and we do not make any distiction between 'primary' and 'derived'
    parameters.
    
    Returns a Pandas DataFrame containing the chain, and two parameters
    containing the number of walkers and the number of iterations in the 
    chain
    """
    
    return 0
    

def analyse_mcmc(chain_file="chain.default", n_burn=50,
                 corner_plot_file="corner.default.png", maxlike_values=None):

    
    
    return 0
