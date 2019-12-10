#!/usr/bin/env python
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
import time
import sys
from ConfigParser import ConfigParser

import numpy as np
import pandas as pd
from scipy import optimize
import emcee
import corner
import matplotlib.pyplot as plt


import hodmodel
import clustering


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
        log10Mmin, log10M1, alpha, log10siglogM, log10M0 = hod_params
        new_hod = hodmodel.HODModel(hod_type=2, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha,
                                    siglogM=10**log10siglogM,
                                    mass_0=10**log10M0)
    elif hod_type == 3:
        log10Mmin, log10M1, alpha, log10siglogM = hod_params
        new_hod = hodmodel.HODModel(hod_type=3, mass_min=10**log10Mmin,
                                    mass_1=10**log10M1, alpha=alpha,
                                    siglogM=10**log10siglogM)
    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    return new_hod


def ndim_from_hod_type(hod_type=1):
    """
    Function to obtain the number of dimensions (parameters) for the HOD
    part of the fit, depending on the HOD type.
    For the moment only hod_type=1 (Kravtsov), hod_type=2 (Zheng) and
    hod_type=3 (Zheng - only 4 parameters) are
    implemented.
    """

    if hod_type == 1:
        return 3
    elif hod_type == 2:
        return 5
    elif hod_type == 3:
        return 4
    else:
        raise RuntimeError("Only implemented values of HOD_type are 1, 2 or 3")


def wp_hod(rp, fit_params, clustobj=None, hod_type=1, fit_f_gal=False,
           fit_gamma=False, nr=100, pimin=0.001, pimax=400, npi=100):
    """
    Basic function to compute wp(rp) for likelihood computations.

    Will use the parameters of the model defined in 'clustobj', which is an
    instance of a HODClustering class, except for the parameters of the HOD,
    and potentially of the ModifiedNFW profile, which we assume that will be
    changing between likelihood calls.
    Mass parameters in HOD are assumed to be given as log10(M_x).
    Parameter f_gal is also assumed to be given as log10(f_gal).

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class).
    The parameters fit_f_gal, fit_gamma define whether we are fitting for
    the respective parameters (and thus their values are included in
    fit_params). If none of these are fitted for, we just keep the default
    profile in clustobj, which we assume will be a NFW.

    Together with the wp, we also return the galaxy density for this model,
    so that it can be also used for the likelihood.
    """

    # First, figure out what are the contents of fit_params
    n_dim_hod = ndim_from_hod_type(hod_type)
    n_dim_prof = fit_f_gal + fit_gamma
    assert len(fit_params) == n_dim_hod + n_dim_prof

    # First, define the new HOD given the parameters
    hod_params = fit_params[:n_dim_hod]
    new_hod = get_hod_from_params(hod_params, hod_type)

    # Now, update the hodclustering object for the new HOD parameters
    clustobj.update_hod(new_hod)

    # If needed, also update the hodclustering object for the new profile
    # parameters
    if n_dim_prof > 0:
        if fit_f_gal:
            # The parameter we actually fit is log10(f_gal)
            f_gal = 10**fit_params[n_dim_hod]
        else:
            f_gal = 1.0   # NFW value

        if fit_gamma:
            gamma = fit_params[-1]
        else:
            gamma = 1.0   # NFW value

        clustobj.update_profile_params(f_gal, gamma)

    # And compute the wp values (use default values for the details)
    return clustering.get_wptotal(
        rpvals=rp, clustering_object=clustobj, nr=nr, pimin=pimin,
        pimax=pimax, npi=npi), clustobj.gal_dens


def chi2_fullmatrix(data_vals, inv_covmat, model_predictions):
    """
    Given a set of data points, its inverse covariance matrix and the
    corresponding set of model predictions, computes the standard chi^2
    statistic (using the full covariances)
    """

    y_diff = data_vals - model_predictions
    return np.dot(y_diff, np.dot(inv_covmat, y_diff))


def lnprior_flat(fit_params, param_lims, hod_type=1, fit_f_gal=False,
                 fit_gamma=False):
    """
    Returns the (un-normalised) log(P) for a flat prior on the HOD parameters
    and, if needed, on the ModNFW profile parameters.
    Mass parameters and sigma_logM are assumed to be given as log10(M_x)
    (so the prior will be flat on the latter), alpha is assumed to be given
    directly.
    For the profile parameters, f_gal is assumed to be given as log10(f_gal),
    while gamma is assumed to be given directly.

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class), and to decide whether
    f_gal and gamma are included through the use of fit_f_gal, fit_gamma.
    """

    # First, figure out what are the contents of fit_params
    # and of param_lims
    n_dim_hod = ndim_from_hod_type(hod_type)
    n_dim_prof = fit_f_gal + fit_gamma
    assert len(fit_params) == n_dim_hod + n_dim_prof
    assert len(param_lims) == 2*len(fit_params)

    # Get the part of the prior corresponding to the HOD parameters
    hod_params = fit_params[:n_dim_hod]
    hod_param_lims = param_lims[:2*n_dim_hod]

    if hod_type == 1:
        logMmin, logM1, alpha = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max = hod_param_lims

        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max:
            lnprior_hod = 0.0
        else:
            lnprior_hod = -np.inf

    elif hod_type == 2:
        logMmin, logM1, alpha, logsiglogM, logM0 = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max, logsiglogM_min, logsiglogM_max, \
            logM0_min, logM0_max = hod_param_lims

        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max and \
           logsiglogM_min < logsiglogM < logsiglogM_max and \
           logM0_min < logM0 < logM0_max:
            lnprior_hod = 0.0
        else:
            lnprior_hod = -np.inf

    elif hod_type == 3:
        logMmin, logM1, alpha, logsiglogM = hod_params
        logMm_min, logMm_max, logM1_min, logM1_max, \
            alpha_min, alpha_max, \
            logsiglogM_min, logsiglogM_max = hod_param_lims

        if logMm_min < logMmin < logMm_max and \
           logM1_min < logM1 < logM1_max and \
           alpha_min < alpha < alpha_max and \
           logsiglogM_min < logsiglogM < logsiglogM_max:
            lnprior_hod = 0.0
        else:
            lnprior_hod = -np.inf

    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    # Get the part of the prior corresponding to the ModNFW profile parameters
    lnprior_prof = 0.0
    if fit_f_gal:
        log_fgal = fit_params[n_dim_hod]
        log_fgal_min, log_fgal_max = param_lims[2*n_dim_hod:(2*n_dim_hod)+2]
        if log_fgal_min < log_fgal < log_fgal_max:
            lnprior_prof += 0.0
        else:
            lnprior_prof += -np.inf

    if fit_gamma:
        gamma = fit_params[-1]
        gamma_min, gamma_max = param_lims[-2:]
        if gamma_min < gamma < gamma_max:
            lnprior_prof += 0.0
        else:
            lnprior_prof += -np.inf

    # Return the total prior
    return lnprior_hod + lnprior_prof


def lnlikelihood_fullmatrix(fit_params, rp, wp, wp_icov, clustobj=None,
                            hod_type=1, fit_f_gal=False, fit_gamma=False,
                            nr=100, pimin=0.001, pimax=400,
                            npi=100, fit_density=0, data_dens=None,
                            data_dens_err=None, data_logdens=None,
                            data_logdens_err=None):
    """
    Computes the (un-normalised) log-likelihood of the data wp(rp) given
    its inverse covariance matrix, and the given values of the HOD and
    profile parameters.

    The rest of the parameters of the model are set in 'clustobj', which
    is an instance of the HODClustering class.

    We assume the errors on wp come from a multi-dimensional Gaussian, and
    use the full covariances.

    We allow the use of different HOD parameterisations using the 'hod_type'
    parameter (same options as in HODModel class), and to indicate which
    of the profile parameters are fitted for.

    We allow also for the inclusion of the galaxy density as part of the
    likelihood computation. The way in which this is done is controlled by
    the fit_density parameter:
        - fit_density = 0: Do not use galaxy density in the computation
        - fit_density = 1: Include gal. density in the computation, assuming
            a Gaussian distribution in density. Need to include parameters
            data_dens and data_dens_err
        - fit_density = 2: Include gal. density in the computation, assuming
            a Gaussin distribution in *log(density)*. Need to include
            parameters data_logdens and data_logdens_err
    """

    wp_model, model_gal_dens = wp_hod(rp=rp, fit_params=fit_params,
                                      clustobj=clustobj, hod_type=hod_type,
                                      fit_f_gal=fit_f_gal, fit_gamma=fit_gamma,
                                      nr=nr, pimin=pimin, pimax=pimax, npi=npi)

    if fit_density == 0:
        chi2_density = 0

    elif fit_density == 1:
        chi2_density = pow((model_gal_dens-data_dens) / data_dens_err, 2)

    elif fit_density == 2:
        model_logdens = np.log10(model_gal_dens)
        chi2_density = pow((model_logdens-data_logdens) / data_logdens_err, 2)

    else:
        raise ValueError("Allowed values of fit_density are 0, 1, 2")

    return -0.5*(chi2_fullmatrix(
        data_vals=wp, inv_covmat=wp_icov, model_predictions=wp_model) +
        chi2_density)


def lnposterior(fit_params, rp, wp, wp_icov, param_lims, clustobj=None,
                hod_type=1, fit_f_gal=False, fit_gamma=False,
                nr=100, pimin=0.001, pimax=400, npi=100,
                fit_density=0, data_dens=None, data_dens_err=None,
                data_logdens=None, data_logdens_err=None):
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

    lp = lnprior_flat(fit_params, param_lims, hod_type, fit_f_gal, fit_gamma)

    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlikelihood_fullmatrix(fit_params, rp, wp, wp_icov,
                                            clustobj, hod_type, fit_f_gal,
                                            fit_gamma, nr, pimin,
                                            pimax, npi, fit_density, data_dens,
                                            data_dens_err, data_logdens,
                                            data_logdens_err)


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


def find_best_fit(fit_params_start, rp, wp, wp_icov, param_lims,
                  return_model=False, minim_method='Powell',
                  minim_options={'xtol': 1e-12, 'ftol': 1e-12},
                  clustobj=None,
                  hod_type=1, fit_f_gal=False, fit_gamma=False,
                  nr=100, pimin=0.001, pimax=400, npi=100,
                  fit_density=0, data_dens=None, data_dens_err=None,
                  data_logdens=None, data_logdens_err=None):
    """
    Function to obtain the best-fit values of the HOD (+ profile) parameters,
    obtaining the maximum of the posterior function (equivalent to the maximum
    likelihood result when using flat priors).

    The maximization is done using standard scipy.optimize functions.
    'minim_method' defines the actual method used, and 'minim_options'
    define additional options for the minimizer (these will be passed to
    optimize.minimize).

    The output of the function is defined by return_model:
    * If return_model==False:
        Just return a tuple containing the HOD (+ profile) parameter values
        for the best fit
    * If return_model==True:
        Returns (fit_params_best, best_model), where best_model is a tuple
        containing some characteristics of the best-fit model:
        (wp_array, galaxy_density, mean_halo_mass, mean_galaxy_bias, frac_sat)
    """

    # First, define the function to minimize (= -log-posterior)
    neglogposterior = lambda *args: -lnposterior(*args)

    # Now, do the actual minimization calling the function
    maxpost_result = \
        optimize.minimize(fun=neglogposterior, x0=fit_params_start,
                          args=(rp, wp, wp_icov, param_lims, clustobj,
                                hod_type, fit_f_gal, fit_gamma,
                                nr, pimin, pimax, npi, fit_density,
                                data_dens, data_dens_err, data_logdens,
                                data_logdens_err),
                          method=minim_method,
                          options=minim_options)

    # Actually print the results
    print "Results of the maximization of the log(Posterior):"
    print maxpost_result

    # Get the best parameters values for output
    fit_params_best = maxpost_result['x']
    minim_result_message = maxpost_result['message']

    # Now, if needed, get other results for this model
    if return_model:
        wp_best, galdens_best = wp_hod(rp, fit_params_best, clustobj,
                                       hod_type, fit_f_gal, fit_gamma,
                                       nr, pimin, pimax, npi)

        # Get the part of the parameters that actually correspond to HOD
        # parameters
        n_dim_hod = ndim_from_hod_type(hod_type)
        hod_params_best = fit_params_best[:n_dim_hod]

        # Define HOD object for the best fit, and initialise the mass
        # array using the clustobj mass array.
        # TODO: can we refactor to avoid this? Do we really need a mass
        #       array associated to an HOD object?
        hod_best = get_hod_from_params(hod_params_best, hod_type)
        hod_best.set_mass_arrays(logM_min=clustobj.logM_min,
                                 logM_max=clustobj.logM_max,
                                 logM_step=clustobj.logM_step)
        meanhalomass_best = \
            hodmodel.mean_halo_mass_hod_array(hod_instance=hod_best,
                                              halo_instance=clustobj.halomodel)
        meangalbias_best = \
            hodmodel.bias_gal_mean_array(hod_instance=hod_best,
                                         halo_instance=clustobj.halomodel)

        fracsat_best = \
            hodmodel.fraction_satellites_array(hod_instance=hod_best,
                                               halo_instance=clustobj.halomodel)

        return fit_params_best, (wp_best, galdens_best,
                                 meanhalomass_best, meangalbias_best,
                                 fracsat_best), minim_result_message

    else:
        return fit_params_best, minim_result_message


def get_initial_walker_positions(n_dimensions=3, n_walkers=100, init_type=0,
                                 param_lims=[-1, 1, -1, 1, -1, 1],
                                 central_position=[0, 0, 0],
                                 ball_size=[0.1, 0.1, 0.1]):
    """
    Function to get the initial positions of walkers in parameter space.
    I implement (for now?) two different recipes, to be chosen by parameter
    'init_type':

    - init_type=0: distribute initial points uniformly inside parameter limits
                   (nothing fancy, so basically a flat prior on the parameters
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
        # TODO: use function for this provided in recent versions of emcee
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
             fit_f_gal=False, fit_gamma=False,
             nr=100, pimin=0.001, pimax=400, npi=100,
             init_type=0, cent_pos=None, ball_size=None,
             n_walkers=100, n_steps_per_walker=100, n_threads=1,
             out_chain_file="chain.default", fit_density=0, data_dens=None,
             data_dens_err=None, data_logdens=None, data_logdens_err=None):
    """
    Function to run the basic MCMC from emcee given the data, and the
    parameters defining the model.

    TODO: description of input parameters and options

    TODO: implement option to return also derived quantities for each sample
          in the chain

    TODO: obtain also relevant properties of the chains to diagnose
          convergence (acceptance fraction, acorr, etc.)
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
        n_dim_hod = 3
        header_hod = "walker logMmin logM1 alpha "

    elif hod_type == 2:
        n_dim_hod = 5
        header_hod = "walker logMmin logM1 alpha logsiglogM logM0 "

    elif hod_type == 3:
        n_dim_hod = 4
        header_hod = "walker logMmin logM1 alpha logsiglogM "
    else:
        raise ValueError("The HOD parameterisation with"
                         "hod_type = %d has not yet been implemented!"
                         % hod_type)

    # Depending on profile parameters to fit, get additional no. of dimensions
    # and additional columns in header
    n_dim_prof = 0
    header_prof = ""
    if fit_f_gal:
        n_dim_prof += 1
        header_prof += "log_fgal "
    if fit_gamma:
        n_dim_prof += 1
        header_prof += "gamma "

    n_dimensions = n_dim_hod + n_dim_prof
    header = header_hod + header_prof + "log_posterior\n"

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
                                    hod_type, fit_f_gal, fit_gamma,
                                    nr, pimin, pimax, npi,
                                    fit_density, data_dens, data_dens_err,
                                    data_logdens, data_logdens_err))

    # And iterate the sampler, writing each of the samples to the output
    # chain file
    for result in sampler.sample(initial_positions,
                                 iterations=n_steps_per_walker,
                                 storechain=False):
        position = result[0]
        lnprob = result[1]

        with open(out_chain_file, "a") as f:
            for k in range(position.shape[0]):
                f.write("%d  %s  %g\n" %
                        (k, np.array_str(position[k],
                                         max_line_width=1000)[1:-1],
                         lnprob[k]))

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

    # First, read in file to DataFrame
    df = pd.read_csv(inchain_file, delim_whitespace=True)

    # Check that there's a column containing the walker no. in the file.
    # Otherwise, there may be a problem with the format
    if 'walker' not in df.columns:
        raise RuntimeError("There may be a problem with the format of file" +
                           inchain_file + ": no column for the walker no.!")

    # Get no. of walkers
    assert df['walker'].min() == 0
    n_walkers = int(df['walker'].max() + 1)

    # And get no. of iterations from the total no. of samples
    n_samples = len(df)

    n_iterations = n_samples/n_walkers

    # Check there was no problem
    assert n_samples == (n_iterations*n_walkers)

    # Everything OK, return results
    return df, n_walkers, n_iterations


def analyse_mcmc(chain_file="chain.default", n_burn=50,
                 corner_plot_file="corner.default.png",
                 perc_intervals=[68.3], maxlike_values=None,
                 plot_quantiles=True, Verbose=False):
    """
    Function to do a basic analysis of a emcee MCMC chain, previously saved
    in a file, following the format of run_mcmc.

    It will first remove the 'burn in' samples (need to give this as an input),
    then creates a corner plot showing the distribution of all the parameters
    stored in the chain.

    Finally, for each parameter it returns the median values and the (2d)
    size of the confidence intervals of the confidence intervals corresponding
    to the perc_intervals.
    By default, these correspond to 1-sigma interval, but several intervals
    can be obtained at once.
    These are returned in a dictionary that contains a list of values for each
    of the parameter in the chain.

    TODO: better description of input parameters and options

    TODO: add option to do analysis only for a subset of the parameters

    TODO: add option to use personalised labels in the plot (e.g. to use
          more appropriate names including LaTeX for parameters)
    """

    # First of all, read data from file
    df_chain, n_walkers, n_iter = read_chain_file(chain_file)

    # Now, remove the burn-in period, and drop the 'walker' column we do not
    # need anymore
    df_chain = df_chain[n_walkers*n_burn:]
    df_chain.drop('walker', axis=1, inplace=True)
    df_chain.drop('log_posterior', axis=1, inplace=True)

    # First, define the 'partial percentiles' corresponding to the intervals
    # we have defined
    perc_intervals = np.atleast_1d(perc_intervals)
    n_int = len(perc_intervals)
    assert (perc_intervals > 0).all()
    assert (perc_intervals < 100).all()
    partial_percents = np.empty((n_int, 2), float)

    for i, pi in enumerate(perc_intervals):
        tail_fract = (100. - pi)/2.
        partial_percents[i, 0] = tail_fract
        partial_percents[i, 1] = 100. - tail_fract

    # Use these to decide the quantiles to include in the plot
    # (if we want to plot them!)
    if plot_quantiles:
        quant_plot = np.concatenate((partial_percents.flatten()/100., [0.5]))
    else:
        quant_plot = []

    # Create the corner plot, and save it to a file (name given as input)
    # We do the "fancy" plot (1-sigma and 2-sigma contours, etc.) as we assume
    # a recent version of the corner package
    # of triangle module >=0.2.0
    fig = corner.corner(df_chain, truths=maxlike_values,
                        quantiles=quant_plot, verbose=False,
                        fill_contours=True, show_titles=True,
                        plot_datapoints=True, levels=perc_intervals/100.)

    fig.savefig(corner_plot_file)

    # Now, compute the dictionary containing the characterisation of the
    # confidence intervals for each of the parameters
    dict_output = {}

    for param_name in df_chain.columns:

        # First, compute the median
        med_value = np.median(df_chain[param_name])
        dict_output[param_name] = [med_value]

        # Now, the lower and upper extents for each of the confidence intervals
        for p in partial_percents:
            extrema = np.percentile(df_chain[param_name], p)
            ci_limits = [med_value - extrema[0], extrema[1] - med_value]
            dict_output[param_name].append(ci_limits)

    # In the Verbose case, also print out the output
    if Verbose:
        print_conf_interval(dict_output, perc_intervals)

    return dict_output


def print_conf_interval(ci_dictionary, perc_intervals=None,
                        file_object=sys.stdout):
    """
    Function to print in a 'nice' way the confidence interval(s) computed
    by function analyse_mcmc.
    We assume the input dictionary has the format as the output of that
    function.
    Will print to a file object that should be already open for
    writting/appending, and that should be closed elsewhere (default is
    sys.stdout)
    """

    if perc_intervals is not None:
        n_int = len(perc_intervals)
    else:
        n_int = len(ci_dictionary[ci_dictionary.keys()[0]]) - 1

    file_object.write("~~~~~~~~\n")
    for param_name in ci_dictionary.keys():
        file_object.write("Parameter: %s\n" % param_name)
        for i in range(n_int):
            if perc_intervals is not None:
                int_string = "%.1f %% interval for parameter: " %\
                    perc_intervals[i]
            else:
                int_string = "Interval for parameter: "

            file_object.write(int_string + "%.6g (-%.2g, +%.2g)\n" %
                              (ci_dictionary[param_name][0],
                               ci_dictionary[param_name][i+1][0],
                               ci_dictionary[param_name][i+1][1]))
        file_object.write("~~~~~~~~\n")

    return 0


def compare_mcmc_data(rp, wp, wperr, n_samples_plot=50,
                      chain_file="chain.default",
                      plot_file="compplot.default.png",
                      n_burn=50, maxlike_values=None, n_params=None,
                      rp_ext=None, wp_ext=None, wperr_ext=None,
                      clustobj=None, hod_type=1, fit_f_gal=False,
                      fit_gamma=False, nr=100, pimin=0.001,
                      pimax=400, npi=100):
    """
    Draw a plot of wp(rp) comparing the data with a sampling of the allowed
    models from the posterior.

    TODO: explain in detail the input parameters and options
    """

    # First, read in the chain samples from the file
    df_chain, n_walkers, n_iter = read_chain_file(chain_file)

    # Remove burn-in period and drop the 'walker' column
    df_chain = df_chain[n_walkers*n_burn:]
    df_chain.drop('walker', axis=1, inplace=True)
    df_chain.drop('log_posterior', axis=1, inplace=True)

    # Define total number of samples in the 'clean' chain
    n_samples_total = n_walkers*(n_iter - n_burn)

    # Decide which rp values will be used for the models (if there is
    # 'extended' data, extend the model also!)
    if rp_ext is None:
        rp_models = rp
    else:
        rp_models = rp_ext

    # Decide on number of parameters to use:
    # By default all parameters in chain, but we can set it to a different
    # number, in case the chain also includes derived parameters
    if n_params is None:
        n_params = len(df_chain.columns)

    # Create figure & axis so that we can start plotting
    fig, ax = plt.subplots()

    # First, plot the models sampled from the chain
    for params in \
        df_chain.values[np.random.randint(n_samples_total,
                                          size=n_samples_plot), :n_params]:

        ax.plot(rp_models,
                wp_hod(rp=rp_models, fit_params=params, clustobj=clustobj,
                       hod_type=hod_type, fit_f_gal=fit_f_gal,
                       fit_gamma=fit_gamma, nr=nr, pimin=pimin, pimax=pimax,
                       npi=npi)[0],
                color='k', alpha=0.1)

    # Now, if given, plot the maximum-likelihood model
    if maxlike_values is not None:
        ax.plot(rp_models,
                wp_hod(rp=rp_models, fit_params=maxlike_values,
                       clustobj=clustobj, hod_type=hod_type,
                       fit_f_gal=fit_f_gal, fit_gamma=fit_gamma, nr=nr,
                       pimin=pimin, pimax=pimax, npi=npi)[0],
                'b-', lw=2, label='Best fit')

    # If present, plot 'extended data'
    if rp_ext is not None:
        ax.errorbar(rp_ext, wp_ext, wperr_ext, marker='s', color='black',
                    linestyle='', label="Extended data")

    # Now, plot the 'real data'
    ax.errorbar(rp, wp, wperr, marker='o', color='red',
                linestyle='', label='Data')

    # Set properties of plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r_p (h^{-1}\, \mathrm{Mpc})$')
    ax.set_ylabel(r'$w_p (h^{-1}\, \mathrm{Mpc})$')

    ax.legend(loc=0)

    # And save figure to file
    fig.savefig(plot_file)

    return 0


def diagnose_plot_chain(chain_file="chain.default",
                        diag_plot_prefix="diagnosechain",
                        n_burn=None, maxlike_values=None):
    """
    Create a 'diagnose plot' showing the evolution of the chain for all
    the walkers for each of the parameters. This is useful, e.g. to decide
    on the number of burn-in steps we want to cut out for later analysis.

    TODO: add option to use personalised labels in the plot (e.g. to use
          more appropriate names including LaTeX for parameters)
    """

    # First, read in the chain samples from the file
    df_chain, n_walkers, n_iter = read_chain_file(chain_file)

    # Group by walker, as we want to plot the evolution of each walker
    # individually
    df_chain_group = df_chain.groupby('walker')

    # Get list of parameters to consider
    param_list = df_chain.columns
    param_list = param_list.drop('walker')

    # And now, do the plot for each of the parameters in the chain
    for j, parameter in enumerate(param_list):
        outfile = diag_plot_prefix + "." + parameter + ".png"

        fig, ax = plt.subplots()

        for i in range(n_walkers):
            walker = df_chain_group.get_group(i)
            ax.plot(range(n_iter), walker[parameter], 'k-', lw=0.5, alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel(parameter)

        # For the case of plotting the log_posterior,
        # change Y axis limits to not beeing too affected by initial steps
        if parameter == "log_posterior":
            max_lp = df_chain[parameter].max()
            ax.set_ylim(max_lp - 20, max_lp + 2)

        # For 'regular' parameters, plot ML values if given
        elif maxlike_values is not None:
            ax.hlines(maxlike_values[j], 0, n_iter, lw=2, color='blue')

        if n_burn is not None:
            ymin, ymax = ax.get_ylim()
            ax.vlines(n_burn, ymin, ymax, lw=2, linestyles='dashed',
                      color='red')
            ax.set_ylim(ymin, ymax)

        ax.grid(True)

        fig.savefig(outfile)

    return 0


def main(paramfile="hodfit_params_default.ini", output_prefix="default"):
    """
    Function to do the full process to fit a HOD model to wp data. It will
    basically consists on running the previous functions in order :P

    Will read all the parameters from an .ini file using the ConfigParser.

    TODO: add options as new features are implemented in the rest of functions
    """

    # First of all, read in the configuration file
    config = ConfigParser()
    config.read(paramfile)

    # Define files for different outputs
    f_config_out = output_prefix + "_config.ini"    # For backup of config
    f_results_out = output_prefix + "_results.txt"  # For summary of results and log of run
    f_chain_out = output_prefix + "_chain.dat"   # For MCMC chain
    f_corner_out = output_prefix + "_corner.png"  # For output corner plot
    f_diag_prefix = output_prefix + "_diagplot"   # Prefix for 'diagnose' plots
    f_compplot_out = output_prefix + "_comparedata.png"  # For plot comparing to data

    # Save a backup of the config file
    with open(f_config_out, 'w') as config_out:
        config.write(config_out)

    # Write initial basic info. to results file
    with open(f_results_out, 'w') as res_out:
        res_out.write("Running the HODfit main function\n")
        res_out.write("Original configuration file: %s\n" % paramfile)
        res_out.write("Backup of the configuration saved to %s\n"
                      % f_config_out)
        res_out.write("This run started at: " + time.asctime() + "\n")
        res_out.write("-------------------------------------------\n")

    # Read in data, select the scales we are interested in, etc.
    infile_wp = config.get('Data', 'wpfile')
    rp, wp, wperr = np.loadtxt(infile_wp, usecols=range(3), unpack=True)
    infile_covmat = config.get('Data', 'wp_cov_file')
    covmatrix = np.loadtxt(infile_covmat)

    intconst = config.getfloat('Data', 'integral_constraint')
    wp = wp + intconst

    rpmin = config.getfloat('Data', 'rpmin')
    rpmax = config.getfloat('Data', 'rpmax')

    rpsel, wpsel, wperrsel, covmatsel = select_scales(rpmin, rpmax, rp, wp,
                                                      wperr, covmatrix)

    # Decide whether to use the full matrix or just the diagonal
    use_full_matrix = config.getboolean('Data', 'use_full_covmatrix')
    if not use_full_matrix:
        covmatsel = np.diag(covmatsel.diagonal())

    # Invert the covariance matrix (selected)
    icovmat_sel = np.linalg.inv(covmatsel)

    # Read in options and data related to galaxy density
    fit_density = config.getint('DataDens', 'use_gal_dens')
    data_dens = None
    data_dens_err = None
    data_logdens = None
    data_logdens_err = None

    if fit_density == 0:
        pass
    elif fit_density == 1:
        data_dens = config.getfloat('DataDens', 'gal_dens')
        data_dens_err = config.getfloat('DataDens', 'gal_dens_error')
    elif fit_density == 2:
        data_logdens = config.getfloat('DataDens', 'gal_logdens')
        data_logdens_err = config.getfloat('DataDens', 'gal_logdens_error')
    else:
        raise ValueError("Allowed values of use_gal_dens are 0, 1 or 2")

    # Read in the parameters defining the HOD model to fit
    hod_type = config.getint('HODModel', 'HOD_type')
    if hod_type == 1:
        n_dim_hod_model = 3
        logMmin_init = config.getfloat('HODModel', 'logMmin_init')
        logM1_init = config.getfloat('HODModel', 'logM1_init')
        alpha_init = config.getfloat('HODModel', 'alpha_init')
        hod_param_init = [logMmin_init, logM1_init, alpha_init]

        logMmin_lims = map(float,
                           config.get('HODModel', 'logMmin_limits').split())
        logM1_lims = map(float, config.get('HODModel', 'logM1_limits').split())
        alpha_lims = map(float, config.get('HODModel', 'alpha_limits').split())
        hod_param_lims = logMmin_lims + logM1_lims + alpha_lims

    elif hod_type == 2:
        n_dim_hod_model = 5
        logMmin_init = config.getfloat('HODModel', 'logMmin_init')
        logM1_init = config.getfloat('HODModel', 'logM1_init')
        alpha_init = config.getfloat('HODModel', 'alpha_init')

        logSiglogM_init = config.getfloat('HODModel', 'logSiglogM_init')
        logM0_init = config.getfloat('HODModel', 'logM0_init')

        hod_param_init = [logMmin_init, logM1_init, alpha_init,
                          logSiglogM_init, logM0_init]

        logMmin_lims = map(float,
                           config.get('HODModel', 'logMmin_limits').split())
        logM1_lims = map(float, config.get('HODModel', 'logM1_limits').split())
        alpha_lims = map(float, config.get('HODModel', 'alpha_limits').split())

        logSiglogM_lims = map(float,
                              config.get('HODModel', 'logSiglogM_limits').split())
        logM0_lims = map(float, config.get('HODModel', 'logM0_limits').split())

        hod_param_lims = logMmin_lims + logM1_lims + alpha_lims +\
            logSiglogM_lims + logM0_lims

    elif hod_type == 3:
        n_dim_hod_model = 4
        logMmin_init = config.getfloat('HODModel', 'logMmin_init')
        logM1_init = config.getfloat('HODModel', 'logM1_init')
        alpha_init = config.getfloat('HODModel', 'alpha_init')

        logSiglogM_init = config.getfloat('HODModel', 'logSiglogM_init')

        hod_param_init = [logMmin_init, logM1_init, alpha_init,
                          logSiglogM_init]

        logMmin_lims = map(float,
                           config.get('HODModel', 'logMmin_limits').split())
        logM1_lims = map(float, config.get('HODModel', 'logM1_limits').split())
        alpha_lims = map(float, config.get('HODModel', 'alpha_limits').split())

        logSiglogM_lims = map(float,
                              config.get('HODModel', 'logSiglogM_limits').split())

        hod_param_lims = logMmin_lims + logM1_lims + alpha_lims + \
            logSiglogM_lims

    else:
        raise ValueError("Allowed values of HOD_type are 1, 2 or 3")

    # Read in the parameters defining the possible additional fit to the
    # profile parameters
    n_dim_prof_model = 0
    prof_param_init = []
    prof_param_lims = []

    fit_f_gal = config.getboolean('ModNFWModel', 'fit_f_gal')
    if fit_f_gal:
        n_dim_prof_model += 1
        log_fgal_init = config.getfloat('ModNFWModel', 'log_fgal_init')
        log_fgal_lims = map(float,
                            config.get('ModNFWModel', 'log_fgal_limits').split())
        prof_param_init += [log_fgal_init]
        prof_param_lims += log_fgal_lims

    fit_gamma = config.getboolean('ModNFWModel', 'fit_gamma')
    if fit_gamma:
        n_dim_prof_model += 1
        gamma_init = config.getfloat('ModNFWModel', 'gamma_init')
        gamma_lims = map(float,
                         config.get('ModNFWModel', 'gamma_limits').split())
        prof_param_init += [gamma_init]
        prof_param_lims += gamma_lims

    # Put together all the parameters that we will try to fit
    n_dim_model = n_dim_hod_model + n_dim_prof_model
    fit_param_init = hod_param_init + prof_param_init
    fit_param_lims = hod_param_lims + prof_param_lims

    # Read in parameters related to the Cosmology to be used, and to
    # details of how to do the calculations
    redshift = config.getfloat('Cosmology', 'redshift')
    omega_matter = config.getfloat('Cosmology', 'omega_matter')
    omega_lambda = config.getfloat('Cosmology', 'omega_lambda')
    pk_lin_z0_file = config.get('Cosmology', 'pk_linear_z0')
    pk_matter_z_file = config.get('Cosmology', 'pk_matter_z')

    logMmin = config.getfloat('HaloModelCalc', 'logMmin')
    logMmax = config.getfloat('HaloModelCalc', 'logMmax')
    logMstep = config.getfloat('HaloModelCalc', 'logMstep')
    halo_exclusion_model = config.getint('HaloModelCalc',
                                         'halo_exclusion_model')

    # Read in file containing pre-computed Fourier profiles,
    # if not valid, set variable to None and will do direct calculation
    fourier_prof_grid_file = config.get('HaloModelCalc',
                                        'fourier_prof_grid_file')
    try:
        fprof_grid_data = np.load(fourier_prof_grid_file)
        fprof_grid_log_krvir = fprof_grid_data['log10_k_rvir']
        fprof_grid_log_conc = fprof_grid_data['log10_concentration']
        fprof_grid_gamma = fprof_grid_data['gamma']
        fprof_grid_profile = fprof_grid_data['profile_grid']
        fprof_grid_rhos = fprof_grid_data['rho_s_unit']
    except:
        if fit_gamma:
            print("Warning: file " + fourier_prof_grid_file +
                  " not valid for pre-computed profile,"
                  " will do direct calculation.")
        fprof_grid_log_krvir = None
        fprof_grid_log_conc = None
        fprof_grid_gamma = None
        fprof_grid_profile = None
        fprof_grid_rhos = None

    # Read in parameters related to the calculation of wp in the models
    # We also define here the r-array needed to define the HODClustering object
    wpcalc_nr = config.getint('WpCalc', 'nr')
    wpcalc_npi = config.getint('WpCalc', 'npi')
    wpcalc_pimin = config.getfloat('WpCalc', 'pimin')
    wpcalc_pimax = config.getfloat('WpCalc', 'pimax')

    rmin = rpsel.min()
    rmax = np.sqrt((wpcalc_pimax**2.) + (rpsel.max()**2.))

    # Define the 'HODClustering' object which we use through.
    # This basically defines the cosmology+halo model+functional form of HOD+
    # numerical properties of the way in which we integrate over M_halo
    #
    # Will use the defaults for many parameters (actual HOD parameters are not
    # relevant here)
    # We always set here the ModNFW parameter to the NFW case (and will
    # modify these later if needed)
    hod_clust =\
        clustering.hod_from_parameters(redshift=redshift, OmegaM0=omega_matter,
                                       OmegaL0=omega_lambda,
                                       powesp_matter_file=pk_matter_z_file,
                                       powesp_linz0_file=pk_lin_z0_file,
                                       hod_type=hod_type, f_gal=1.0,
                                       gamma=1.0, logM_min=logMmin,
                                       logM_max=logMmax, logM_step=logMstep,
                                       rmin=rmin, rmax=rmax, nr=wpcalc_nr,
                                       rlog=True,
                                       halo_exclusion_model=halo_exclusion_model,
                                       fprof_grid_log_krvir=fprof_grid_log_krvir,
                                       fprof_grid_log_conc=fprof_grid_log_conc,
                                       fprof_grid_gamma=fprof_grid_gamma,
                                       fprof_grid_profile=fprof_grid_profile,
                                       fprof_grid_rhos=fprof_grid_rhos)

    # Now, we start the fun! First, *if required*, get the best-fit model using
    # Scipy minimisation methods
    do_best_fit_minimization = \
        config.getboolean('BestFitcalc', 'do_best_fit_minimization')

    if do_best_fit_minimization:
        bestfit_params, bestfit_derived, bestfit_message =\
            find_best_fit(fit_params_start=fit_param_init, rp=rpsel, wp=wpsel,
                          wp_icov=icovmat_sel, param_lims=fit_param_lims,
                          return_model=True, clustobj=hod_clust, hod_type=hod_type,
                          fit_f_gal=fit_f_gal, fit_gamma=fit_gamma,
                          nr=wpcalc_nr, npi=wpcalc_npi, pimin=wpcalc_pimin,
                          pimax=wpcalc_pimax, fit_density=fit_density,
                          data_dens=data_dens, data_dens_err=data_dens_err,
                          data_logdens=data_logdens,
                          data_logdens_err=data_logdens_err)

        # Get goodness of fit for this
        chi2_bestfit = chi2_fullmatrix(data_vals=wpsel, inv_covmat=icovmat_sel,
                                       model_predictions=bestfit_derived[0])
        ndof = len(rpsel) - n_dim_model

        # Add part coming from galaxy density, if needed
        if fit_density == 1:
            chi2_dens = pow((bestfit_derived[1]-data_dens) / data_dens_err, 2)
            chi2_bestfit += chi2_dens
            ndof += 1
        elif fit_density == 2:
            model_logdens = np.log10(bestfit_derived[1])
            chi2_dens = pow((model_logdens-data_logdens) / data_logdens_err, 2)
            chi2_bestfit += chi2_dens
            ndof += 1

        # Write results to 'results' file
        with open(f_results_out, 'a') as res_out:
            res_out.write("BEST-FIT MODEL:\n")
            res_out.write("Best-fit calculation finished at: " + time.asctime() + "\n")
            res_out.write("Minimization termination message: " + bestfit_message + "\n")
            res_out.write("Best-fit parameters: " + str(bestfit_params) + "\n")
            res_out.write("Chi^2 = %.5g\n" % chi2_bestfit)
            res_out.write("Number of degrees of freedom, ndof = %d\n" % ndof)
            res_out.write("Chi^2/ndof = %.3g\n" % (chi2_bestfit/ndof))
            res_out.write("Derived parameters: \n")
            res_out.write("  Galaxy density = %.5g\n" % bestfit_derived[1])
            res_out.write("  Mean halo mass = %.5g\n" % bestfit_derived[2])
            res_out.write("  Mean galaxy bias = %.5g\n" % bestfit_derived[3])
            res_out.write("  Satellite fraction = %.5g\n" % bestfit_derived[4])
            res_out.write("-------------------------------------------\n")

    else:
        bestfit_params = None

    # Now, do the actual MCMC run to get the sample chain
    # First, read in relevant parameters and decide on the way to initialise
    # the walkers
    n_walkers = config.getint('MCMCcalc', 'number_walkers')
    n_iterations = config.getint('MCMCcalc', 'number_iterations')
    n_threads = config.getint('MCMCcalc', 'threads')

    mcmc_init_type = config.getint('MCMCcalc', 'mcmc_init_type')

    if mcmc_init_type == 0:
        cpos = None
        ball_size = None
    elif mcmc_init_type == 1:
        if do_best_fit_minimization:
            cpos = bestfit_params
        else:
            cpos = fit_param_init
        ball_size = map(float,
                        config.get('MCMCcalc', 'mcmc_init_ball').split())
        assert len(ball_size) == n_dim_model

    else:
        raise ValueError("Allowed values of mcmc_init_type are 0 or 1")

    # Now, actually run the MCMC
    run_mcmc(rp=rpsel, wp=wpsel, wp_icov=icovmat_sel,
             param_lims=fit_param_lims, clustobj=hod_clust, hod_type=hod_type,
             fit_f_gal=fit_f_gal, fit_gamma=fit_gamma,
             nr=wpcalc_nr, pimin=wpcalc_pimin, pimax=wpcalc_pimax,
             npi=wpcalc_npi, init_type=mcmc_init_type, cent_pos=cpos,
             ball_size=ball_size, n_walkers=n_walkers,
             n_steps_per_walker=n_iterations, n_threads=n_threads,
             out_chain_file=f_chain_out, fit_density=fit_density,
             data_dens=data_dens, data_dens_err=data_dens_err,
             data_logdens=data_logdens, data_logdens_err=data_logdens_err)

    # Once the MCMC run is finished, do a basic analysis to get constraints
    # on parameters
    # Note this will depend on the burn in period we assume, better to play
    # with this taking into account the actual chain
    n_burn_in = config.getint('MCMCanalysis', 'burn_in_iterations')
    ci_percent = map(float,
                     config.get('MCMCanalysis', 'conf_intervals').split())

    mcmc_analysis_result = analyse_mcmc(
        chain_file=f_chain_out, n_burn=n_burn_in,
        corner_plot_file=f_corner_out, perc_intervals=ci_percent,
        maxlike_values=bestfit_params)

    # Write results to 'results' file
    with open(f_results_out, 'a') as res_out:
        res_out.write("MCMC SAMPLING OF THE POSTERIOR:\n")
        res_out.write("Full sample chain written to file %s\n" % f_chain_out)

    with open(f_results_out, 'a') as res_out:
        print_conf_interval(ci_dictionary=mcmc_analysis_result,
                            perc_intervals=ci_percent,
                            file_object=res_out)

    with open(f_results_out, 'a') as res_out:
        res_out.write("-------------------------------------------\n")

    # Now, get other plots for assessment of the results
    nsamples_comp = config.getint('MCMCanalysis', 'n_samp_comparison_plot')
    compare_mcmc_data(rp=rpsel, wp=wpsel, wperr=wperrsel,
                      n_samples_plot=nsamples_comp, chain_file=f_chain_out,
                      plot_file=f_compplot_out, n_burn=n_burn_in,
                      maxlike_values=bestfit_params, rp_ext=rp, wp_ext=wp,
                      wperr_ext=wperr, clustobj=hod_clust, hod_type=hod_type,
                      fit_f_gal=fit_f_gal, fit_gamma=fit_gamma,
                      nr=wpcalc_nr, pimin=wpcalc_pimin, pimax=wpcalc_pimax,
                      npi=wpcalc_npi)

    diagnose_plot_chain(chain_file=f_chain_out, diag_plot_prefix=f_diag_prefix,
                        n_burn=n_burn_in, maxlike_values=bestfit_params)

    # Write final blob to results file
    with open(f_results_out, 'a') as res_out:
        res_out.write("Corner plot for the posterior distribution of "
                      "the parameters saved to %s\n" % f_corner_out)
        res_out.write("Plot comparing samples to data saved to %s\n"
                      % f_compplot_out)
        res_out.write("Plots showing the evolution of the chains drawn "
                      "to files starting by %s\n" % f_diag_prefix)
        res_out.write("-------------------------------------------\n")
        res_out.write("This run finished at: " + time.asctime() + "\n")

    return 0


# If running the script from command line, run the main function!
if __name__ == "__main__":

    main(*sys.argv[1:])

    sys.exit(0)
