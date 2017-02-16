#!/usr/bin/env python

# import hods_clustering as hc
# import hods_hodmodel as hod
from hodfit import clustering as hc
from hodfit import hodmodel as hod
import numpy as np
import time

redshift = 0.52
OmegaM_0 = 0.274
OmegaL_0 = 1. - OmegaM_0
pkfile_lin_z0 = "WMAP7_linz0_matterpower.dat"
pkfile_matter_z = "WMAP7_z0_matterpower.dat"

hodclust_object = hc.hod_from_parameters(redshift=redshift, OmegaM0=OmegaM_0,
                                         OmegaL0=OmegaL_0, powesp_matter_file=pkfile_matter_z,
                                         powesp_linz0_file=pkfile_lin_z0, logM_min=6.,
                                         logM_max=17., logM_step=0.01)


NRCALC = 100
NPICALC = 100
PIMAXCALC = 350.0

def wp_hod(rp, log10Mmin, log10M1, alpha, hodclustering):
    new_hod = hod.HODModel(hod_type=1, mass_min = 10**log10Mmin, mass_1 = 10**log10M1, alpha=alpha)
    hodclustering.update_hod(new_hod)
    return hc.get_wptotal(rpvals=rp, clustering_object=hodclustering, nr=NRCALC, npi=NPICALC, pimax=PIMAXCALC)

logMmin_start = 11.5
logM1_start = 13.0
alpha_start = 1.2

rp = np.logspace(np.log10(0.3),np.log10(12),16)

rmin = rp.min()
rmax = np.sqrt(PIMAXCALC**2 + rp.max()**2)
rarray = np.logspace(np.log10(rmin), np.log10(rmax), NRCALC)
hodclust_object.update_rvalues(rarray)


t1 = time.time()
wp_start = wp_hod(rp=rp, log10Mmin=logMmin_start, log10M1=logM1_start, alpha=alpha_start, hodclustering=hodclust_object)
t2 = time.time()

print "Computing wp took ", t2 - t1, " seconds."


hodclust_object = hc.hod_from_parameters(redshift=redshift, OmegaM0=OmegaM_0,
                                         OmegaL0=OmegaL_0, powesp_matter_file=pkfile_matter_z,
                                         powesp_linz0_file=pkfile_lin_z0, logM_min=6.,
                                         logM_max=17., logM_step=0.01, halo_exclusion_model=1)

hodclust_object.update_rvalues(rarray)


t1 = time.time()
wp_start = wp_hod(rp=rp, log10Mmin=logMmin_start, log10M1=logM1_start, alpha=alpha_start, hodclustering=hodclust_object)
t2 = time.time()

print "Computing wp *with the simple halo exclusion (=1)* took ", t2 - t1, " seconds."


hodclust_object = hc.hod_from_parameters(redshift=redshift, OmegaM0=OmegaM_0,
                                         OmegaL0=OmegaL_0, powesp_matter_file=pkfile_matter_z,
                                         powesp_linz0_file=pkfile_lin_z0, logM_min=6.,
                                         logM_max=17., logM_step=0.01, halo_exclusion_model=0)

hodclust_object.update_rvalues(rarray)

t1 = time.time()
wp_start = wp_hod(rp=rp, log10Mmin=logMmin_start, log10M1=logM1_start, alpha=alpha_start, hodclustering=hodclust_object)
t2 = time.time()

print "Computing wp *without halo exclusion* took ", t2 - t1, " seconds."

# Add possibility to change the profile
def wp_hod_fgal_gamma(rp, log10Mmin, log10M1, alpha, fgal, gamma, hodclustering):
    new_hod = hod.HODModel(hod_type=1, mass_min = 10**log10Mmin, mass_1 = 10**log10M1, alpha=alpha)
    hodclustering.update_hod(new_hod)
    hodclustering.update_profile_params(fgal, gamma)
    return hc.get_wptotal(rpvals=rp, clustering_object=hodclustering, nr=NRCALC, npi=NPICALC, pimax=PIMAXCALC)

fgal_test = 1.1
gamma_test = 1.1


hodclust_object = hc.hod_from_parameters(redshift=redshift, OmegaM0=OmegaM_0,
                                         OmegaL0=OmegaL_0, powesp_matter_file=pkfile_matter_z,
                                         powesp_linz0_file=pkfile_lin_z0, logM_min=6.,
                                         logM_max=17., logM_step=0.01)
hodclust_object.update_rvalues(rarray)

t1 = time.time()
wp_start = wp_hod_fgal_gamma(rp=rp, log10Mmin=logMmin_start, log10M1=logM1_start,
                             alpha=alpha_start, fgal=fgal_test, gamma=gamma_test,
                             hodclustering=hodclust_object)
t2 = time.time()

print "Computing wp *for the ModNFW* and *with default halo exclusion (=2)* took ", t2 - t1, " seconds."

hodclust_object = hc.hod_from_parameters(redshift=redshift, OmegaM0=OmegaM_0,
                                         OmegaL0=OmegaL_0, powesp_matter_file=pkfile_matter_z,
                                         powesp_linz0_file=pkfile_lin_z0, logM_min=6.,
                                         logM_max=17., logM_step=0.01, halo_exclusion_model=1)

hodclust_object.update_rvalues(rarray)           

t1 = time.time()
wp_start = wp_hod_fgal_gamma(rp=rp, log10Mmin=logMmin_start, log10M1=logM1_start,
                             alpha=alpha_start, fgal=fgal_test, gamma=gamma_test,
                             hodclustering=hodclust_object)
t2 = time.time()

print "Computing wp *for the ModNFW* and *with simple halo exclusion (=1)* took ", t2 - t1, " seconds."                                                      