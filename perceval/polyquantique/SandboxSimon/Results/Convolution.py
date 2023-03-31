
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from dataRead import *
from Functions import *


def photonFun(x,gam,x0,sig):
    #sig = 18e-12 + 9e-12 + 10e-12
    f =  1/2 * np.exp(gam / 2 * ( gam * sig ** 2 - 2 * x + 2 * x0)) * gam * (
        sc.special.erfc((gam * sig **2 - x + x0)/(np.sqrt(2)*sig))
    )
    f /= np.sqrt(Overlap(f,f,x))
    return f


def createWaves(dataPath,doYouPlot = False):


    # Importation des données et un peu de traitement
    time, env1, env2 ,delay, table = xpRead(dataPath)
    idx = np.argmax(env1)
    interestingRange = [int(idx-200),int(idx+200)]

    realTime = np.array(time[interestingRange[0]:interestingRange[1]],dtype = 'float64')
    exTime = np.linspace(realTime[0],realTime[-1],10000)/1e12  

    #Starting values of the fit
    gamTest = 1e10
    x0Test = exTime[idx]
    sigTest = 37e-12
    waves = []

    for env in [env1,env2]:

        realEnv = np.array(env[interestingRange[0]:interestingRange[1]])
        realEnvInterp = np.interp(exTime,realTime/1e12,realEnv)
        realEnvInterp /= np.sqrt(Overlap(realEnvInterp,realEnvInterp,exTime))
        popt, pcov = sc.optimize.curve_fit(photonFun,exTime,realEnvInterp,p0 = [gamTest,x0Test,sigTest])


        waves.append(Exponential(exTime-popt[1],popt[0]))
    waves = np.array(waves)
    if doYouPlot:
        plt.figure()
        plt.plot(waves.T)

    return exTime, waves, delay, table
