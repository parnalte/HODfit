#!/usr/bin/env python

#import hods_clustering as hc
from hodfit import clustering as hc
import numpy as np
import pylab as pl

hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01)

rp = np.logspace(-2,2,30)

print "Computing default model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)

#With constant bias
hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01, scale_dep_bias=False)

print "Computing constant bias model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp_cbias = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)

#Without Mvir limit in cs term
hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01, use_mvir_limit=False)

print "Computing no-Mvir-lim model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp_nomvirlim = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)

#Without halo exclusion in 2h term
hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01, halo_exclusion_model=0)

print "Computing no halo exclusion model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp_nohaloexcl = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)

#With Zheng2004's model for  halo exclusion in 2h term
hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01, halo_exclusion_model=1)

print "Computing Zheng2004 halo exclusion model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp_zhenghaloexcl = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)

#With original MoWhite bias parameters
hcobj = hc.hod_from_parameters(redshift=0.496412, OmegaM0=0.27, OmegaL0=0.73, powesp_linz0_file="WMAP7_linz0_matterpower.dat", powesp_matter_file="WMAP7_z0p52_matterpower.dat", hod_type=2, hod_mass_min=10**11.906, hod_mass_1=10**13.285, hod_alpha=1.4091, hod_siglogM=0.48456, hod_mass_0=10**6.6762, logM_step=0.01, use_tinker_bias_params=False)

print "Computing original bias model"
hcobj.update_rvalues(np.logspace(-2,2.8,300))
wp_origbias = hc.get_wptotal(rpvals=rp, clustering_object=hcobj)


rpcp, wpcp = np.loadtxt("haloplot_wp.out", usecols=range(2), unpack=1)

pl.plot(rpcp, wpcp, 'k-', lw=3, label="CosmoPMC")
pl.plot(rp,wp,'o-', lw=2, label="My code - Present full model")
pl.plot(rp,wp_cbias, '+-', label="My code - Constant bias")
pl.plot(rp,wp_nomvirlim, '+-', label="My code - No Mvir limit in CS term")
pl.plot(rp,wp_nohaloexcl, '+-', label="My code - No halo exclusion in 2h term")
pl.plot(rp,wp_zhenghaloexcl, '+-', label="My code - Zheng2004 model for halo exlusion in 2h term")
pl.plot(rp, wp_origbias, '+-', label="My code - Original bias parameters from MoWhite2002")




pl.loglog()
pl.legend(loc=0)
pl.xlabel(r'$r_p$')
pl.ylabel(r'$w_p$')
pl.show()
