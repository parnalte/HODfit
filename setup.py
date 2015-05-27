# -*- coding: utf-8 -*-
"""
Basic setup for package hodfit.
Basis taken from http://www.scotttorborg.com/python-packaging/minimal.html

Created on Thu May 21 15:57:59 2015
@author: pablo

TODO:
    - Add dependencies
    - Add docs
    - Add tests
"""

from setuptools import setup

setup(name='hodfit',
      version='0.4',
      description='Fit HOD models to wp(rp) data',
      url='https://bitbucket.org/parnalte/hod-fit',
      author='Pablo Arnalte-Mur',
      author_email='pablo.arnalte@uv.es',
      license='BSD',
      packages=['hodfit'],
      scripts=['bin/hodfit-full'],
      install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'emcee',
                        'astropy', 'hankel'],
      zip_safe=False)
