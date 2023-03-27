import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.optimize import root


def Exp(x, gamma):
    """Calculates an Exponential enveloppe for an array of positions.

    Args:
        x (array): positions.
        gamma (float): exponential parameter.

    Returns:
        (array): Exponential values at x."""

    E = (gamma) ** (1 / 2) * np.exp(-gamma * x / 2) * np.heaviside(x, 1)

    return E

def Lrz(x,gamma):
    L = np.sqrt(2*gamma/np.pi)/(gamma-2*1j*(x))
    return L

"""YOU HAVE TO DEFINE THE SHIFT t0 AND THE KNOWN HOM COEFFICIENT hom_experimental"""
c=299792458 #speed of light [m/s]
t0 = 30e-12 #shift in time [s]
z0 =c*t0 #shift in space [m]
hom_experimental=0.93



def HOM(Tau):
    N=100000

    gamma = 1/(c*Tau*1e-12)
    lam = 500*1e-9 #wave length
    k0 = 2*np.pi/lam #mean k vector
    z = np.linspace(-5/gamma,20/gamma, N)
    k = np.linspace(k0-1000*gamma,k0+1000*gamma,N)

    

    env1 = Exp(z,gamma)
    env2 = Exp(z-z0,gamma)
    env1_k = Lrz(k-k0,gamma)
    env2_k = np.exp(1j*k*z0)*env1_k

    #plt.figure()
    #plt.plot(z,env1)
    #plt.plot(z,env2)
    #plt.show()

    # Beam splitter transfer matrix
    BS=1/np.sqrt(2) * np.array([[1,-1],[1,1]])
    A = BS[0,0]; B = BS[0,1]; C = BS[1,0]; D = BS[1,1]

    overlap = abs(sc.integrate.simps(env1_k*np.conj(env2_k),k))**2
    Pr11 = abs(A)**2*abs(D)**2 + abs(B)**2*abs(C)**2 + 2*np.real(np.conj(A)*np.conj(D)*B*C)*overlap 
    coeff_HOM=1-2*Pr11

    return coeff_HOM-hom_experimental

sol=sc.optimize.bisect(HOM,1,1000)
print(sol)





 

