import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def Gaussian(x, s):
    """Calculates a Gaussian enveloppe for an array of positions.

    Args:
        x (array): positions.
        s (float): standard deviation.

    Returns:
        (array): Gaussian values at x."""

    G = 1 / np.sqrt(s * np.sqrt(2 * np.pi)) * np.exp(-(x**2) / (4 * s**2))

    return G

def Lrz(x,gamma):
    L = np.sqrt(2*gamma/np.pi)/(gamma-2*1j*(x))
    return L


def Exponential(x, gamma):
    """Calculates an Exponential enveloppe for an array of positions.

    Args:
        x (array): positions.
        gamma (float): exponential parameter.

    Returns:
        (array): Exponential values at x."""

    E = (gamma) ** (1 / 2) * np.exp(-gamma * x / 2) * np.heaviside(x, 1)

    return E

def Overlap(u, v, x):
    """Calculates the overlap integral between 2 wave functions.

    Args:
        u (array): wave function 1.
        v (array): wave function 2..
        x (array): positions.

    Returns:
        (float): Overlap integral value."""

    OverLap = sc.integrate.simps(np.conj(u) * v, x)

    return OverLap

def Schmidt(waves,x):
    """Computes an othonormal basis based on a non-orhtogonal set of wave functions, using the Gram-Schmidt method.

    Args:
        waves (array): non orthogonal set of wave functions.
        x (array): position array.

    Returns:
        new_base (array): orthonormal basis.
        coeffs (array): matrix of projection coefficients."""

    N_shift = np.shape(waves)[0]
    N = len(x)
    new_base = np.zeros((N_shift, N))

    for i in range(N_shift):
        new_base[i] = waves[i]
        for j in range(i):
            new_base[i] = new_base[i] - Overlap(new_base[j], waves[i], x) * new_base[j]
        new_base[i] = new_base[i] / np.sqrt(Overlap(new_base[i], new_base[i], x))
    
    coeffs=np.zeros((N_shift,N_shift))
    for i in range(N_shift):
        for j in range(i+1):
            coeffs[i,j]=Overlap(waves[i], new_base[j], x)

    return new_base, coeffs

def Lowdin(waves,x):
    """Computes an othonormal basis based on a non-orhtogonal set of wave functions, using the Lowdin method.

    Args:
        waves (array): non orthogonal set of wave functions.
        x (array): position array.

    Returns:
        new_base (array): orthonormal basis.
        coeffs (array): matrix of projection coefficients."""

    N_shift = np.shape(waves)[0]
    N = len(x)
    new_base = np.zeros((N_shift, N))

    # Overlap matrix S
    S = np.zeros((N_shift, N_shift))
    for i in range(N_shift):
        S[i, i] = 1
        for j in range(i):
            S[i, j] = Overlap(waves[i], waves[j], x)
            S[j, i] = S[i, j]

    # Finding eigenvalues + eigenvectors
    eigenvals, U = np.linalg.eigh(S)

    S_diag = np.eye(N_shift)
    for i in range(N_shift):
        S_diag[i, i] = abs(eigenvals[i])

    # Calculating new wave functions
    U = np.matrix(U)
    S_sqrt = np.matmul(np.matmul(U, np.sqrt(S_diag)), U.H)
    S_sqrt_inv = np.linalg.inv(S_sqrt)
    new_base = np.array(np.matmul(S_sqrt_inv, waves))
    coeffs = S_sqrt
    return new_base, coeffs









