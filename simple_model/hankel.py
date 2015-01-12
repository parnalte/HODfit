
"""
    hankel.py --- Utilities for the Hankel transform (useful when transforming P(k) <--> xi(r))

    This piece of code is taken directly from the CosmoPy package (v. 0.64), by Mark Neyrinck,
    Istvan Szapudi, Adrian Pope, Peter Papai, and Tamas Budavari.

    I may want to implement my own version of the transforms.
"""



import numpy as N
import bessint as BI

class Hankel:
    def __init__(self,dim=3,nMax=32768):
        """ 
        dim=3 for 3D P(k) <-> xi(r)
        dim=2 for 2D
        """
        self.dim = dim
        if (self.dim == 3):
            self.bessint = BI.BesselIntegrals(0.5,nMax)
        elif (self.dim == 2):
            self.bessint = BI.BesselIntegrals(0.,nMax)
        else:
            print 'dim has to be 2 or 3'

    def transform1(self,f,x,n,h,pk2xi=True):
        """
        Does the transform on a single x
        """
        if (self.dim == 3):
            bi = 1.0/x**3*self.bessint.besselInt(lambda z:z**1.5*f(z/x),n,h)
            pf = 1.0/(2.0*N.pi)**1.5
        else:
            bi = 1.0/x**2*self.bessint.besselInt(lambda z:z*f(z/x),n,h)
            pf = 1.0/(2.0*N.pi)
        if not(pk2xi):
            pf =1.0/pf
        return pf*bi

    def transform(self,f,x,n=32768,h=1./512.,pk2xi=True):
        """
        f is a function giving p(k), if transforming to xi(r), or
        xi(r), if transforming to p(k).  If you're transforming data,
        that means you should call a function that interpolates that
        data, e.g. utils.splineIntLinExt_LogLog().  The function will
        generally be evaluated well outside the range where it's
        interpolating, so be sure that it extrapolates sensibly,
        e.g. lim x->inf f(x) = 0.

        x is the array of abscissas of the result, e.g. if
        p(k)->xi(r), x will be the array of points where xi is to be
        evaluated.

        n is the number of Bessel-function zeroes to include.  As suggested,
        1000 probably will work, but test this.

        h is the stepsize of the integration. 1/512 seems to work.
        """
        if N.isscalar(x):
            return self.transform1(f,x,n,h,pk2xi)
        else:
            return N.array(map(lambda z:self.transform1(f,z,n,h,pk2xi),x))
