#! /usr/bin/python
#! -*- coding: utf-8 -*-

##****************************************************************##
"""
   Programa: 'llibreria_bias.py'
   Autor: Pablo Arnalte Mur
   Data creacio: 12/07/07
   Data ultima modificacio: 19/07/07

   Funcions per calcular el bias en funció de la massa dels halos
   i del redshift, a partir de les fòrmules de
   Mo & White (MNRAS 336:112, 2002).
   Emprarem els paràmetres cosmològics emprant en la simulació de
   Heinamaki et al.:

   \Omega_M = 0.27
   \Omega_{\Lambda} = 0.73
   h = 0.71
   \sigma_8 = 0.84

   Variables globals:
   Omega_M0 = 0.27 --> densitat de matèria actual
   Omega_L0 = 0.73 --> densitat d'energia fosca actual
   h = 0.71 --> ct. de Hubble
   A = 2.1907e+6 --> normalització de l'espectre de potències
   delta_c = 1.69 --> constant en les definicions de nu
"""
##*****************************************************************##


__all__ = ["D","P","sigma","bias", "bias_alt"]

##LLIBRERIES QUE IMPORTEM
from scipy import *
from scipy.integrate import quad, Inf
from pylab import load, save


Omega_M0 = 0.27
Omega_L0 = 0.73
h = 0.71
A = 2.1907e+6
delta_c = 1.69

##Critical density in units of h^2 M_sol Mpc^-3
rho_crit = 2.7752e+11


###Part d'evolució amb redshift
def _E(z):
    Wm = Omega_M0*((1+z)**3.)
    return sqrt(Omega_L0 + Wm)

def _Wm(z):
    eixida = Omega_M0*((1+z)**3.)/(_E(z)**2.)
    return eixida

def _WL(z):
    eixida = Omega_L0/(_E(z)**2.)
    return eixida

def _g(z):
    denominador = (_Wm(z)**(4./7.)) - _WL(z) + ((1+(_Wm(z)/2.))*(1+(_WL(z)/70.)))
    eixida = 5.*_Wm(z)/(2.*denominador)
    return eixida

def D(z):
    """Calcula el factor de creiximent per a fluctuacions linials,
       a partir dels paràmetres emprats en la simulació, i seguint les fòrmules (10) i (11)
       de Mo&White.
    """
    eixida = _g(z)/(_g(z=0.)*(1+z))
    return eixida

##Part de espectres de potències, sigma(r), i relacions M<-->R
def _R(M):
    #rho0 = Omega_M0*2.77536627e+11
    rho0 = Omega_M0*rho_crit
    eixida = (3*M/(4*pi*rho0))**(1./3.)
    return eixida

def _T(k):
    q = k/(Omega_M0*h)
    eixida = log(1+2.34*q)*((1 + 3.89*q + ((16.1*q)**2.) + ((5.46*q)**3.) + ((6.71*q)**4.))**(-1./4.))/(2.34*q)
    return eixida

def P(k):
    """Calcula l'espectre de potències per a l'Univers de la nostra simulació.
       Calculat a partir de l'equació (7) de Mo&White, i normalitzat a partir de (6).

       k = unitats de h Mpc-1
       P = l'eixida tindrà unitats de h-3 Mpc3
    """   
    return A*k*_T(k)*_T(k)

def _W(x):
    return 3*(sin(x) - (x*cos(x)))/(x**3.)

def _integrand(k,Rad):
    y = k*Rad
    eixida = (k**2.)*P(k)*(_W(y)**2.)
    return eixida


def sigma(Rad):
    """Calcula la sigma de les fluctuacions en massa convolucionades amb un filtre 'top-hat' esfèric
       de radi 'Rad'.

       Rad = unitats de h-1 Mpc
    """
    integral = quad(lambda x: _integrand(k=x,Rad=Rad), a=0., b=Inf, limit = 300, epsabs=5.0e-5, epsrel=5.0e-5)
    eixida = sqrt(integral[0] / (2*pi*pi))
    return eixida


##Part del bias en funció de M i z
def _nu(M,z):
    R_loc = _R(M)
    D_loc = D(z)
    sigma_loc = sigma(R_loc)
    eixida = delta_c / (D_loc*sigma_loc)
    return eixida

def bias(M,z):
    """Calcula el bias en funció de M i z segons la fòrmula (17) de Mo&White.
       La massa ha d'anar donada en unitats de h-1 M_sun
    """   
    nuloc = _nu(M,z)
    eixida = 1 + ((nuloc*nuloc - 1)/delta_c)
    return eixida

def bias_alt(M,z):
    """Calcula el bias en funció de M i z segons la fòrmula (19) de Mo&White.
       La massa ha d'anar donada en unitats de h-1 M_sun
    """
    a = 0.707
    b = 0.5
    c = 0.6
    
    nuloc = _nu(M,z)
    nu_alt = sqrt(a)*nuloc

    eixida = 1 + (((nu_alt**2) + (b*(nu_alt**(2*(1-c)))) - (((nu_alt**(2*c))/sqrt(a))/((nu_alt**(2*c)) + (b*(1-c)*(1 - (c/2))))))/delta_c)
    return eixida

    
##Part sobre mass function/number density
def ndens_differential(M,z):
    """Calculates the 'differential' part of eq. (14) in MoWhite02.
       In order to get the appropriate integral terms, should integrate
       ndens_differential*dnu
    """
    A_par = 0.322
    a_par = 0.707
    q_par = 0.3

    nuloc = _nu(M,z)

    nu_alt = sqrt(a_par)*nuloc

#    rho_mean = _Wm(z)*rho_crit
    rho_mean = Omega_M0*rho_crit

    #The sqrt(a) term comes from the d(nu') term, so that
    #we can do the actual integral over nu
    term1 = sqrt(a_par)*A_par*sqrt(2/pi)
    term2 = 1. + (1./pow(nu_alt, 2*q_par))
    term3 = rho_mean/M
    term4 = exp(-nu_alt*nu_alt/2.)

    return term1*term2*term3*term4

    
def ndens_integral(M_array, z):
    """Basically, just do the integral of the function above, given a range (and binning) in halo mass.
    """

    N_m = len(M_array)
    N_bins = N_m - 1

    suma = 0
    for i in range(N_bins):

        #print i, "/", N_bins

        M_mean = sqrt(M_array[i]*M_array[i+1]) #logarithmic mean
        nu_1 = _nu(M_array[i], z)
        nu_2 = _nu(M_array[i+1], z)

        suma = suma + (ndens_differential(M_mean, z)*(nu_2 - nu_1))

    return suma

    
def mean_bias_ndens(M_array, z):
    """Obtain the mean bias for a given mass range doing the weighted average over the mass function
    """

    N_m = len(M_array)
    N_bins = N_m - 1

    suma = 0
    for i in range(N_bins):

        #print i, "/", N_bins

        M_mean = sqrt(M_array[i]*M_array[i+1]) #logarithmic mean
        nu_1 = _nu(M_array[i], z)
        nu_2 = _nu(M_array[i+1], z)
        bias_bin = bias_alt(M_mean, z)
        
        suma = suma + (bias_bin*ndens_differential(M_mean, z)*(nu_2 - nu_1))

    mean_dens = ndens_integral(M_array, z)
    print mean_dens
    print suma
    mean_bias = suma/mean_dens

    return mean_bias
    

def mean_mass_ndens(M_array, z):
    """Obtain the mean halo mass for a given mass range doing the weighted average over the mass function
    """

    N_m = len(M_array)
    N_bins = N_m - 1

    suma = 0


    for i in range(N_bins):

        #print i, "/", N_bins

        M_mean = sqrt(M_array[i]*M_array[i+1]) #logarithmic mean
        nu_1 = _nu(M_array[i], z)
        nu_2 = _nu(M_array[i+1], z)

        suma = suma + (M_mean*ndens_differential(M_mean, z)*(nu_2 - nu_1))

    mean_dens = ndens_integral(M_array, z)
    print mean_dens
    print suma
    wmean_mass = suma/mean_dens

    return wmean_mass


def combined_integration(M_array, z):
    """Do all the previous integrations together in one go (for speed!)
       Returns: density, mean_bias, mean_mass
    """

    N_m = len(M_array)
    N_bins = N_m - 1

    dens_sum = 0.
    bias_sum = 0.
    meanmass_sum = 0.

    for i in range(N_bins):

        M_mean = sqrt(M_array[i]*M_array[i+1]) #logarithmic mean
        nu_1 = _nu(M_array[i], z)
        nu_2 = _nu(M_array[i+1], z)
        bias_bin = bias_alt(M_mean, z)

        dens_sum = dens_sum + (ndens_differential(M_mean, z)*(nu_2 - nu_1))
        bias_sum = bias_sum + (bias_bin*ndens_differential(M_mean, z)*(nu_2 - nu_1))
        meanmass_sum = meanmass_sum + (M_mean*ndens_differential(M_mean, z)*(nu_2 - nu_1))

    bias_mean = bias_sum/dens_sum
    meanmass  = meanmass_sum/dens_sum

    return dens_sum, bias_mean, meanmass


        




    

    

