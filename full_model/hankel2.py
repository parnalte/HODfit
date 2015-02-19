
"""
Pablo Arnalte-Mur (OAUV-Valencia)

Adapt hankel.py from PIP package 'hankel' to optimise
for the case of SphericalHankelTransform of order 0,
to avoid nested if-else clauses and similar things

"""

import numpy as np
from scipy.special import yv, jv

class SphericalHankelTransform0(object):

    def __init__(self, N=200, h=0.05):
        nu=0.5
        self._nu = nu
        self._h = h
        self._zeros = self._roots(N)
        self.x = self._x(h)
        self.j = self._j(self.x)
        self.w = self._weight()
        self.dpsi = self._d_psi(h * self._zeros)

    def _f(self, f, x):
        return np.sqrt(np.pi / (2 * x)) * f(x)

    def _roots(self, N):
        return (np.arange(N) + 1)

    def _psi(self, t):
        print t
        y = np.sinh(t)
        print y
        return t * np.tanh(np.pi * y / 2)

    def _d_psi(self, t):
        a = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (1.0 + np.cosh(np.pi * np.sinh(t)))
        a[np.isnan(a)] = 1.0
        return a

    def _weight(self):
        return yv(self._nu, np.pi * self._zeros) / self._j1(np.pi * self._zeros)

    def _j(self, x):
        return jv(self._nu, x)

    def _j1(self, x):
        return jv(self._nu + 1, x)

    def _x(self, h):
        return np.pi * self._psi(h * self._zeros) / h

    def transform(self, f, ret_err=True, ret_cumsum=False):
        """
        Perform the transform of the function f
        
        Parameters
        ----------
        f : callable
            A function of one variable, representing :math:`f(x)`
            
        ret_err : boolean, optional, default = True
            Whether to return the estimated error
            
        ret_cumsum : boolean, optional, default = False
            Whether to return the cumulative sum
        """
        fres = self._f(f, self.x)
        summation = np.pi * self.w * fres * self.j * self.dpsi
        ret = [np.sum(summation)]
        if ret_err:
            ret.append(summation[-1])
        if ret_cumsum:
            ret.append(np.cumsum(summation))

        return ret




