
"""Test module for hods_halomodel.py
"""


import numpy as np
import astropy.cosmology as ac
from hods_halomodel import *


def test_delta_c_0():
    this_cosmo = ac.WMAP7
    this_redshift = 0.0
    result = 1.674
    assert(abs(delta_c_z(redshift=this_redshift, cosmo=this_cosmo) - result) < 0.005)

def test_Dz_0():
    this_redshift = 0.
    this_result = 1.
    assert(growth_factor_linear(this_redshift) == this_result)

