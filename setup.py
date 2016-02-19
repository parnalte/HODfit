# -*- coding: utf-8 -*-
"""
Basic setup for package hodfit.
Basis taken from http://www.scotttorborg.com/python-packaging/minimal.html

Created on Thu May 21 15:57:59 2015
@author: pablo

TODO:
    - Add docs
    - Add tests
"""

from setuptools import setup

setup(name='hodfit',
      version='0.6',
      description='Fit HOD models to wp(rp) data',
      url='https://github.com/parnalte/HODfit',
      author='Pablo Arnalte-Mur',
      author_email='pablo.arnalte@uv.es',
      license='BSD',
      packages=['hodfit'],
      scripts=['bin/hodfit-full'],
      install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'emcee',
                        'astropy', 'hankel'],
      include_package_data=True,
      zip_safe=False)
