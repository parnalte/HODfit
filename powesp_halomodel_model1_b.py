#!/usr/bin/env python
#! -*- coding: utf-8 -*-

"""Following some equations from

   - Abbas et al. (2010), MNRAS, 406:1306
   - Coupon et al. (2012), A&A, 542:A5
   - Cooray & Sheth (2002), Phys. Rep. 372:1

   Some stuff taken from my old 'llibreria_bias', and modified as needed.

"""



import numpy as np
import scipy.integrate as SI
import scipy.special as SS
from sys import argv, exit


#Global variables
Omega_M0 = 0.3
Omega_L0 = 0.7
h = 0.7
A = 2.298e+6
delta_c = 1.686
REDSHIFT = 0.0
rho_crit = 2.77536627e11   #Critical density of the universe in units of h^2 M_sun Mpc^-3



###Part d'evolució amb redshift
def _E(z):
    Wm = Omega_M0*((1+z)**3.)
    return np.sqrt(Omega_L0 + Wm)

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
    rho0 = Omega_M0*2.77536627e+11
    eixida = (3*M/(4*np.pi*rho0))**(1./3.)
    return eixida

def _T(k):
    q = k/(Omega_M0*h)
    eixida = np.log(1+2.34*q)*((1 + 3.89*q + ((16.1*q)**2.) + ((5.46*q)**3.) + ((6.71*q)**4.))**(-1./4.))/(2.34*q)
    return eixida

def P(k):
    """Calcula l'espectre de potències per a l'Univers de la nostra simulació.
       Calculat a partir de l'equació (7) de Mo&White, i normalitzat a partir de (6).

       k = unitats de h Mpc-1
       P = l'eixida tindrà unitats de h-3 Mpc3
    """   
    return A*k*_T(k)*_T(k)

def _W(x):
    return 3*(np.sin(x) - (x*np.cos(x)))/(x**3.)

def _integrand(k,Rad):
    y = k*Rad
    eixida = (k**2.)*P(k)*(_W(y)**2.)
    return eixida


def sigma(Rad):
    """Calcula la sigma de les fluctuacions en massa convolucionades amb un filtre 'top-hat' esfèric
       de radi 'Rad'.

       Rad = unitats de h-1 Mpc
    """
    integral = SI.quad(lambda x: _integrand(k=x,Rad=Rad), a=0., b=SI.Inf, limit = 100, epsabs=5.0e-9, epsrel=5.0e-9)
    eixida = np.sqrt(integral[0] / (2*np.pi*np.pi))
    return eixida


def sigma_arr(Rad, klin, pklin):

    x2 = klin[1:]
    x1 = klin[:-1]

    krvar = klin*Rad
    Wvar = _W(krvar)
    integ_array = klin*klin*pklin*Wvar*Wvar

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    integ_result = comb_array.sum()

    result = np.sqrt(integ_result/(2.*np.pi*np.pi))
    return result

    
    

    



    
##Part del bias en funció de M i z
def _nu(M,z):
    R_loc = _R(M)
    D_loc = D(z)
    sigma_loc = sigma(R_loc)
    eixida = delta_c / (D_loc*sigma_loc)
    return eixida

def nu_arr(M, z, klin, pklin):
    R_loc = _R(M)
    #D_loc = D(z)
    #In this case, P(k) given at the appropriate redshift, so sigma(R) already
    #containts term D(z)
    sigma_loc = sigma_arr(R_loc, klin, pklin)
 #   eixida = delta_c / (D_loc*sigma_loc)
    eixida = delta_c/sigma_loc
    return eixida



################
def HOD_function_Z(mass_in, M_min, M1, alpha):
    """HOD function from Zehavi2005, as given in eq. (9) of Abbas2010

       Try to write it in an 'array friendly' way
    """
    mass = np.atleast_1d(mass_in)
    
    result = np.atleast_1d(1 + pow(mass/M1, alpha))
    result[mass<=M_min] = 0

    return result


def HOD_function_Z_CS(mass_in, M_min, M1, alpha):
    """HOD function from Zehavi2005, as given in eq. (9) of Abbas2010,
       but separating the central from the satellites.

       We assume that, all haloes containing galaxies have a central one,
       and the rest are satellites.

       Try to write it in an 'array friendly' way
    """
    mass = np.atleast_1d(mass_in)

    result_c = np.ones(len(mass))
    result_c[mass<=M_min] = 0

    result_s = np.atleast_1d(pow(mass/M1, alpha))
    result_s[mass<=M_min] = 0

    return result_c, result_s
    
    

def HOD_coupon_centrals(mass_in, M_min, sigma_log):
    """HOD for centrals from eq. (11) in Coupon2012
    """

    mass = np.atleast_1d(mass_in)

    result = 0.5*(1. + SS.erf((np.log10(mass) - np.log10(M_min))/sigma_log))
    return result

def HOD_coupon_sats(mass_in, M0, M1, alpha):
    """HOD for satellites from eq. (12) in Coupon2012
    """

    mass = np.atleast_1d(mass_in)

    result = np.atleast_1d(pow((mass - M0)/M1, alpha))
    result[mass<=M0] = 0
    
    return result

    
    

    
    
def Cos_Int(x):
    """Taken from eq. (82) in Cooray2002,
       needed for the NFW profile definition in k-space
    """
    integ = SI.quad(lambda t: 1/t, a=x, b=SI.Inf, limit=300, epsrel=1.0e-8, weight='cos', wvar = 1.0)
    return -integ[0]

def Cos_Int_array(x):

    func = np.vectorize(Cos_Int)
    return func(x)
    

def Sin_Int(x):
    """Taken from eq. (82) in Cooray2002,
       needed for the NFW profile definition in k-space

       Have to do this strange stuff to avoid div_by_zero problems
    """
    if(x==0):
        result = 0
    elif(x>1.0):
        integ1 = SI.quad(lambda t: 1/t, a=0.0, b=0.1, limit=3000, epsrel = 1.0e-6, weight='sin', wvar=1.0)
        integ2 = SI.quad(lambda t: 1/t, a=0.1, b=0.5, limit=3000, epsrel = 1.0e-6, weight='sin', wvar=1.0)
        integ3 = SI.quad(lambda t: 1/t, a=0.5, b=x, limit=3000, epsrel = 1.0e-6, weight='sin', wvar=1.0)
        result = integ1[0] + integ2[0] + integ3[0]
    else:
        integ = SI.quad(lambda t: 1/t, a=0.0, b=x, limit=3000, epsrel = 1.0e-6, weight='sin', wvar=1.0)
        result = integ[0]
    return result

def Sin_Int_array(x):

    func = np.vectorize(Sin_Int)
    return func(x)

    
def u_nfwprof_fourier(k, rs, conc):
    """Directly from eqs. (81) and (76) of Cooray2002
    """
    krvar = k*rs
    
    prefactor = 1./(np.log(1.+conc) - (conc/(1.+conc)))
    term1 = np.sin(krvar)*(Sin_Int_array((1.+conc)*krvar) - Sin_Int_array(krvar))
    term2 = np.sin(conc*krvar)/((1.+conc)*krvar)
    term3 = np.cos(krvar)*(Cos_Int_array((1.+conc)*krvar) - Cos_Int_array(krvar))

    return prefactor*(term1 - term2 + term3)

    
def mass_from_nu(nu_vals, nu_inputarr, mass_inputarr):

    mass_out = np.interp(nu_vals, nu_inputarr, mass_inputarr)
    return mass_out

    
def concentration(mass, mass_star):
    """We follow definition in eq. (A.10) of Coupon2012

    """
    conc_0 = 11.0
    beta = 0.13

    return (conc_0/(1.+REDSHIFT))*pow(mass/mass_star, -beta)


def Delta_vir(redshift):
    """ Equation (A.12) in Coupon2012

    """
    fact1 = 18*np.pi*np.pi
    fact2 = 0.399
    expon = 0.941
    Omz = _Wm(z=redshift)

    result = fact1*(1. + (fact2*pow((1./Omz) - 1., expon)))

    return result

def rvir_from_mass(mass, DDvir):
    """Inversion of equation (A.11) in Coupon2012
    """

    rho_0 = rho_crit*Omega_M0  ##Present-day mean matter density
    result = pow(3.*mass/(4.*np.pi*rho_0*DDvir), 1./3.)

    return result


def mvir_from_radius(radius, DDvir):
    """ Equation (A.11) in Coupon2012
    """
    rho_0 = rho_crit*Omega_M0  ##Present-day mean matter density

    mass = 4.*np.pi*pow(radius, 3)*rho_0*DDvir/3.

    return mass
    

    
def uprof_from_mass(k, mass, Mstar, Delta_vir):

    vir_radius = rvir_from_mass(mass, Delta_vir)
    conc = concentration(mass, Mstar)

    scale_radius = vir_radius/conc

    return u_nfwprof_fourier(k=k, rs=scale_radius, conc=conc)


def rhoprof_NFW(radius, rho_s, rad_s):

    result = rho_s/((radius/rad_s)*pow(1. + (radius/rad_s), 2))
    return result

    
def rhoprof_from_mass(r, mass, Mstar, Delta_vir):
    
    vir_radius = rvir_from_mass(mass, Delta_vir)
    conc = concentration(mass, Mstar)
    scale_radius = vir_radius/conc

    rho_s = (mass/(4.*np.pi*pow(scale_radius, 3)))*pow(np.log(1.+conc) - (conc/(1. + conc)), -1)
    return rhoprof_NFW(radius=r, rho_s = rho_s, rad_s=scale_radius)
    
    
def dist_func_nu(nuval):
    """ Eq. (A.6) in Coupon2012, using values for the parameters
        from Abbas2010 (between eqs. 7 and 8)
    """
    Amp = 0.322
    a = 0.71
    p = 0.3

    result = (Amp/nuval)*np.sqrt(2.*a*nuval*nuval/np.pi)*(1. + pow(a*nuval*nuval, -p))*np.exp(-a*nuval*nuval/2.0)
    return result

def integrand_1halo_term(nuval, kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    mass = mass_from_nu(nu_vals = nuval, nu_inputarr=nu_inputarr, mass_inputarr = mass_inputarr)
    uprof = uprof_from_mass(k=kval, mass=mass, Mstar=mass_star, Delta_vir=delta_vir)
    ng_hod = HOD_function_Z(mass_in = mass, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval = dist_func_nu(nuval=nuval)

    result = fval*ng_hod*(ng_hod - 1)*pow(np.fabs(uprof), 2)/mass
    return meandens*result

def integrand_1halo_term_arr(nu_array, mass_array, kval, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    fval_array = dist_func_nu(nuval=nu_array)
    ng_hod_array = HOD_function_Z(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    uprof_array = uprof_from_mass(k=kval, mass=mass_array, Mstar=mass_star, Delta_vir=delta_vir)

    result = meandens*fval_array*ng_hod_array*ng_hod_array*pow(np.fabs(uprof_array),2)/mass_array

    return result


def integrand_1halo_term_CS_arr(nu_array, mass_array, kval, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):
    
    meandens = Omega_M0*rho_crit
    fval_array = dist_func_nu(nuval=nu_array)
    nc_hod_array, ns_hod_array = HOD_function_Z_CS(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    uprof_array = uprof_from_mass(k=kval, mass=mass_array, Mstar=mass_star, Delta_vir=delta_vir)

    result_c = meandens*fval_array*nc_hod_array*ns_hod_array*np.fabs(uprof_array)/mass_array
    result_s = meandens*fval_array*ns_hod_array*ns_hod_array*pow(np.fabs(uprof_array),2)/mass_array

    return result_c + result_s

    


    
def integral_1halo_term(kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    numin = nu_inputarr[0]
    numax = nu_inputarr[-1]

    intres = SI.quad(lambda x: integrand_1halo_term(x, kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod), a=numin, b=numax, limit=1000, epsrel=1.0e-4)

    return intres[0]

def integral_1halo_term_arr(kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    x2 = nu_inputarr[1:]
    x1 = nu_inputarr[:-1]

    integ_array = integrand_1halo_term_arr(nu_array = nu_inputarr, mass_array=mass_inputarr, kval=kval, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result

    
def integral_1halo_term_CS_arr(kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    x2 = nu_inputarr[1:]
    x1 = nu_inputarr[:-1]

    integ_array = integrand_1halo_term_CS_arr(nu_array = nu_inputarr, mass_array=mass_inputarr, kval=kval, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result
    
    
def integrand_meandensity(nuval, nu_inputarr, mass_inputarr, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    mass = mass_from_nu(nu_vals = nuval, nu_inputarr=nu_inputarr, mass_inputarr = mass_inputarr)
    ng_hod = HOD_function_Z(mass_in = mass, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval = dist_func_nu(nuval=nuval)

    result = meandens*fval*ng_hod/mass
    return result

def integrand_meandensity_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    ng_hod_array = HOD_function_Z(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval_array = dist_func_nu(nuval=nu_array)

    result = meandens*fval_array*ng_hod_array/mass_array
    return result

    
def meandensity(nu_inputarr, mass_inputarr, Mmin_hod, M1_hod, alpha_hod):
    
    numin = nu_inputarr[0]
    numax = nu_inputarr[-1]

    intres = SI.quad(lambda x: integrand_meandensity(x, nu_inputarr, mass_inputarr, Mmin_hod, M1_hod, alpha_hod), a=numin, b=numax, limit=1000, epsrel=1.0e-4)

    return intres[0]
    
def meandensity_arr(nu_inputarr, mass_inputarr, Mmin_hod, M1_hod, alpha_hod):

    x2 = nu_inputarr[1:]
    x1 = nu_inputarr[:-1]
    
    integ_array = integrand_meandensity_arr(nu_array = nu_inputarr, mass_array=mass_inputarr, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result

   

def calc_pk_1haloterm(kvalues_array, Mmin_hod, M1_hod, alpha_hod, mass_star, delta_vir, nu_inputarr, mass_inputarr):

    Nk = len(kvalues_array)
    pk1h_out = np.empty(Nk, float)

    dens_mean = meandensity_arr(nu_inputarr = nu_inputarr, mass_inputarr=mass_inputarr, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    for i in range(Nk):
        print "k-value ", i+1, " of ", Nk

        pk1h_out[i] = integral_1halo_term_arr(kval = kvalues_array[i], nu_inputarr=nu_inputarr, mass_inputarr=mass_inputarr, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    pk1h_out = pk1h_out/(dens_mean*dens_mean)

    return pk1h_out

    
def calc_pk_1haloterm_CS(kvalues_array, Mmin_hod, M1_hod, alpha_hod, mass_star, delta_vir, nu_inputarr, mass_inputarr):

    Nk = len(kvalues_array)
    pk1h_out = np.empty(Nk, float)

    dens_mean = meandensity_arr(nu_inputarr = nu_inputarr, mass_inputarr=mass_inputarr, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    for i in range(Nk):
        print "k-value ", i+1, " of ", Nk

        pk1h_out[i] = integral_1halo_term_CS_arr(kval = kvalues_array[i], nu_inputarr=nu_inputarr, mass_inputarr=mass_inputarr, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    pk1h_out = pk1h_out/(dens_mean*dens_mean)

    return pk1h_out


def bias_func_nu(nuval):
    """ Eq. (6) in Abbas2010
    """

    a = 0.71
    p = 0.3
    result = 1. + ((a*nuval*nuval - 1.)/delta_c) - ((2.*p/delta_c)/(1. + pow(a*nuval*nuval, p)))
    return result


def integrand_2halo_term(nuval, kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    mass = mass_from_nu(nu_vals = nuval, nu_inputarr=nu_inputarr, mass_inputarr = mass_inputarr)
    uprof = uprof_from_mass(k=kval, mass=mass, Mstar=mass_star, Delta_vir=delta_vir)
    ng_hod = HOD_function_Z(mass_in = mass, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval = dist_func_nu(nuval=nuval)
    bias = bias_func_nu(nuval=nuval)

    result = meandens*fval*ng_hod*bias*np.fabs(uprof)/mass
    return result

    
def integrand_2halo_term_arr(nu_array, mass_array, kval, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):
    
    meandens = Omega_M0*rho_crit
    fval_array = dist_func_nu(nuval=nu_array)
    ng_hod_array = HOD_function_Z(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    uprof_array = uprof_from_mass(k=kval, mass=mass_array, Mstar=mass_star, Delta_vir=delta_vir)
    bias_array = bias_func_nu(nuval=nu_array)

    result = meandens*fval_array*ng_hod_array*bias_array*np.fabs(uprof_array)/mass_array
    return result


    
def integral_2halo_term(kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    numin = nu_inputarr[0]
    numax = nu_inputarr[-1]

    intres = SI.quad(lambda x: integrand_2halo_term(x, kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod), a=numin, b=numax, limit=1000, epsrel=1.0e-6)

    return intres[0]


def integral_2halo_term_arr(kval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, M1_hod, alpha_hod):

    x2 = nu_inputarr[1:]
    x1 = nu_inputarr[:-1]

    integ_array = integrand_2halo_term_arr(nu_array = nu_inputarr, mass_array=mass_inputarr, kval=kval, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result



def calc_pkfactor_2haloterm(kvalues_array, Mmin_hod, M1_hod, alpha_hod, mass_star, delta_vir, nu_inputarr, mass_inputarr):

    Nk = len(kvalues_array)
    pkfact2h_out = np.empty(Nk, float)
    dens_mean = meandensity_arr(nu_inputarr = nu_inputarr, mass_inputarr=mass_inputarr, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    for i in range(Nk):
        print "k-value ", i+1, " of ", Nk

        pkfact2h_out[i] = integral_2halo_term_arr(kval = kvalues_array[i], nu_inputarr=nu_inputarr, mass_inputarr=mass_inputarr, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    pkfact2h_out = (pkfact2h_out*pkfact2h_out)/(dens_mean*dens_mean)

    return pkfact2h_out
    


    
def integrand_xir1hCS_arr(nu_array, mass_array, rval, mass_star, delta_vir, Mmin_hod, sigma_hod, M0_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    fval_array = dist_func_nu(nuval=nu_array)
    nc_hod_array = HOD_coupon_centrals(mass_in = mass_array, M_min = Mmin_hod, sigma_log = sigma_hod)
    ns_hod_array = HOD_coupon_sats(mass_in = mass_array, M0=M0_hod, M1=M1_hod, alpha = alpha_hod)
    rhoprof_array = rhoprof_from_mass(r = rval, mass=mass_array, Mstar = mass_star, Delta_vir=delta_vir)

    result = meandens*fval_array*nc_hod_array*ns_hod_array*rhoprof_array/mass_array
    return result

def integral_xir1hCS_arr(rval, nu_inputarr, mass_inputarr, mass_star, delta_vir, Mmin_hod, sigma_hod, M0_hod, M1_hod, alpha_hod):

    #First, have to define Mvir for r, to get the lower limit for the integral
    virial_mass = mvir_from_radius(radius=rval, DDvir=delta_vir)

    imin = np.searchsorted(mass_inputarr, virial_mass)

    nu_rval = nu_inputarr[imin:]
    mass_rval = mass_inputarr[imin:]

    ####
    x2 = nu_rval[1:]
    x1 = nu_rval[:-1]

    integ_array = integrand_xir1hCS_arr(nu_array = nu_rval, mass_array=mass_rval, rval=rval, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, sigma_hod=sigma_hod, M0_hod=M0_hod, M1_hod=M1_hod,alpha_hod=alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]
    
    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result



def calc_xir_1haloCS(rvalues_array, Mmin_hod, sigma_hod, M0_hod, M1_hod, alpha_hod, mass_star, delta_vir, nu_inputarr, mass_inputarr):

    Nr = len(rvalues_array)
    xir_out = np.empty(Nr, float)

    dens_mean = meandensity_arr(nu_inputarr = nu_inputarr, mass_inputarr=mass_inputarr, Mmin_hod=Mmin_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    for i in range(Nr):
        print "r-value ", i+1, " of ", Nr

        xir_out[i] = integral_xir1hCS_arr(rval=rvalues_array[i], nu_inputarr=nu_inputarr, mass_inputarr=mass_inputarr, mass_star=mass_star, delta_vir=delta_vir, Mmin_hod=Mmin_hod, sigma_hod=sigma_hod, M0_hod=M0_hod, M1_hod=M1_hod, alpha_hod=alpha_hod)

    xir_out = (2.*xir_out/(dens_mean*dens_mean)) - 1.0

    return xir_out

    
    
#####Derived quantities

def integrand_meanmass_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod):

    meandens = Omega_M0*rho_crit
    ng_hod_array = HOD_function_Z(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval_array = dist_func_nu(nuval=nu_array)
    result = meandens*fval_array*ng_hod_array

    return result

def integrand_normal(nu_array, mass_array):
    meandens = Omega_M0*rho_crit
    fval_array = dist_func_nu(nuval=nu_array)
    result = meandens*fval_array/mass_array
    return result

def integral_normalization(nu_array, mass_array):

    x2 = nu_array[1:]
    x1 = nu_array[:-1]
    integ_array = integrand_normal(nu_array, mass_array)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result


    
def meanmass_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod):
    """ Following eq. (15) in Coupon2012
    """
    
    #Numerator
    x2 = nu_array[1:]
    x1 = nu_array[:-1]

    integ_array = integrand_meanmass_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]
    comb_array = 0.5*(y2+y1)*(x2-x1)

    numer = comb_array.sum()
    #normal = integral_normalization(nu_array, mass_array)
    normal = meandensity_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod)
    result = numer/normal
    return result
    

def integrand_meanbias_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod):
    meandens = Omega_M0*rho_crit
    ng_hod_array = HOD_function_Z(mass_in = mass_array, M_min = Mmin_hod, M1 = M1_hod, alpha = alpha_hod)
    fval_array = dist_func_nu(nuval=nu_array)
    bias_array = bias_func_nu(nuval=nu_array)

    result = meandens*fval_array*ng_hod_array*bias_array/mass_array
    return result

def meanbias_array(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod):
    """ Following eq. (13) in Coupon2012
    """

        #Numerator
    x2 = nu_array[1:]
    x1 = nu_array[:-1]

    integ_array = integrand_meanbias_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod)

    y2 = integ_array[1:]
    y1 = integ_array[:-1]
    comb_array = 0.5*(y2+y1)*(x2-x1)

    numer = comb_array.sum()
    normal = meandensity_arr(nu_array, mass_array, Mmin_hod, M1_hod, alpha_hod)
    result = numer/normal
    return result


    
def normal_fnu(nu_array):

    x2 = nu_array[1:]
    x1 = nu_array[:-1]
    fval_array = dist_func_nu(nuval=nu_array)

    y2 = fval_array[1:]
    y1 = fval_array[:-1]

    comb_array = 0.5*(y2+y1)*(x2-x1)
    result = comb_array.sum()
    
    return result


    
    



    
    

    

if __name__=="__main__":



    if len(argv)!=6:
        print "Use: ", argv[0], " linear_pk_file  Mmin  M1  alpha  output_prefix\n"
        exit()
    
    linpkfile = argv[1]
    Mmin = float(argv[2])
    M1 = float(argv[3])
    alpha = float(argv[4])
    outprefix = argv[5]


    #Start calculation: first read in P_lin(k)
    klin, pklin = np.loadtxt(linpkfile, usecols=range(2), unpack=True)
    Nk = len(klin)


    ##The first step is to get a 'two-way' relation between 'M' and 'nu'
    ##We can get it from eq. (A.2) of Coupon12, will tabulate it (so we can also get M(nu) doing a simple interpolation)

    #Define the array to use (should expan the full domain for integrations)
#    logM_min = 2.0
#    logM_max = 17.0
#    Msteps = 200

#    logM_min = 1.0
    logM_max = 20.0
    Msteps = 1000

    logM_min = np.log10(Mmin)

    
    M_arr_in = 10**(np.linspace(logM_min, logM_max, Msteps))
    nu_vecfunc = np.vectorize(lambda x: nu_arr(M=x, z=REDSHIFT, klin=klin, pklin=pklin))
    nu_arr_in = nu_vecfunc(M_arr_in)

    #Calculate quantities that depend only on redshift (not halos mass)
    crit_dens_virial = Delta_vir(redshift=REDSHIFT)
    mass_star = np.interp(1.0, nu_arr_in, M_arr_in)


    #################
    #Start calculation: first read in P_lin(k)
    #klin, pklin = np.loadtxt(linpkfile, usecols=range(2), unpack=True)
    #Nk = len(klin)
    
    #Now, calculate the 1halo term:
    pk_1halo = calc_pk_1haloterm_CS(kvalues_array=klin, Mmin_hod=Mmin, M1_hod=M1, alpha_hod=alpha, mass_star=mass_star, delta_vir=crit_dens_virial, nu_inputarr=nu_arr_in, mass_inputarr=M_arr_in)

    #And the 2halo term:
    pk_2halo_factor = calc_pkfactor_2haloterm(kvalues_array=klin, Mmin_hod=Mmin, M1_hod=M1, alpha_hod=alpha, mass_star=mass_star, delta_vir=crit_dens_virial, nu_inputarr=nu_arr_in, mass_inputarr=M_arr_in)

    pk_2halo = pklin*pk_2halo_factor

    #And the total
    pktotal = pk_1halo + pk_2halo

    #And write output
    outfile_1h = outprefix + "_1halo.dat"
    X = np.empty((Nk, 2), float)
    X[:,0] = klin
    X[:,1] = pk_1halo
    np.savetxt(outfile_1h, X, fmt="%.8g")

    outfile_2h = outprefix + "_2halo.dat"
    X = np.empty((Nk, 2), float)
    X[:,0] = klin
    X[:,1] = pk_2halo
    np.savetxt(outfile_2h, X, fmt="%.8g")

    outfile_tot = outprefix + "_total.dat"
    X = np.empty((Nk, 2), float)
    X[:,0] = klin
    X[:,1] = pktotal
    np.savetxt(outfile_tot, X, fmt="%.8g")

    print "Done!"

    
    
    
    

    