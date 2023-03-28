import matplotlib.pyplot as plt
import numpy as np
from Functions import Gaussian, Schmidt


env_type = 1  # 1=Gaussian, 2=exponential
N = 2000  # number of x points to plot the wave functions
N_shift = 2  # total number of shifted wave functions
waves = np.zeros((N_shift, N))  # values for each wave function

sigma_wave = 0.5  # standard deviation of wave function
x = np.linspace(-16 * sigma_wave, 16 * sigma_wave, N)  # position values

shift_list=np.linspace(-8*sigma_wave,8*sigma_wave,500)
C2=np.zeros(len(shift_list))
for k in range(len(shift_list)):
    shift=shift_list[k]
    x_shift=np.array([0,shift])
    for i in range(N_shift):
        waves[i] = Gaussian(x-x_shift[i], sigma_wave)

    new_base,coeffs=Schmidt(waves,x)
    C2[k]=coeffs[1,1]
plt.figure()
plt.plot(shift_list,C2**2/2)
plt.xlabel(r'$\Delta_z$')
plt.ylabel(r'$C_2^2/2$')
plt.show()