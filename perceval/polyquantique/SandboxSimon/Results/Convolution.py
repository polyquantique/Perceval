
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from dataRead import *
from Functions import *


def photonFun(x,gam,x0,sig):
    #sig = 18e-12 + 9e-12 + 10e-12

    """     f = 1/4 * np.exp(gam / 4 * ( gam * sig ** 2 - 4 * x + 4 * x0)) * gam / (np.sqrt(np.pi)*sig) * (
            sc.special.erfc((gam * sig **2 - 2*x + 2 * x0)/(2*sig))
        ) """
    f =  1/2 * np.exp(gam / 2 * ( gam * sig ** 2 - 2 * x + 2 * x0)) * gam * (
        sc.special.erfc((gam * sig **2 - x + x0)/(np.sqrt(2)*sig))
    )
    #f /= np.sqrt(Overlap(f,f,x))
    #f /= sc.integrate.simps(f,x)
    #f /= np.max(np.abs(f))
    return f


def createWaves(dataPath,doYouPlot = False, timeArray = 10000):


    # Importation des données et un peu de traitement
    time, env1, env2 ,delay, table, vHom = xpRead(dataPath)
    idx = np.argmax(env1)
    interestingRange = [int(idx-200),int(idx+300)]
    #interestingRange = [0,-1]


    realTime = np.array(time[interestingRange[0]:interestingRange[1]],dtype = 'float64')
    exTime = np.linspace(realTime[0],realTime[-1],timeArray)/1e12

    #Starting values of the fit
    gamTest = 1e10
    x0Test = exTime[idx]
    sigTest = 37e-12
    ATest = 1
    waves = []
    fullWaves = []
    
    for env in [env1,env2]:

        realEnv = np.array(env[interestingRange[0]:interestingRange[1]])
        realEnvInterp = np.interp(exTime,realTime/1e12,realEnv)
        #realEnvInterp /= np.sqrt(Overlap(realEnvInterp,realEnvInterp,exTime))
        realEnvInterp /= sc.integrate.simps(realEnvInterp,exTime)
        popt, pcov = sc.optimize.curve_fit(photonFun,exTime,realEnvInterp,p0 = [gamTest,x0Test,sigTest])


       # fullWaves.append(photonFun(exTime,*popt))
        fullWaves.append(realEnvInterp)

        waves.append(Exponential(exTime-popt[1],popt[0]))

    if doYouPlot:
        plt.figure()
        plt.plot(np.array(fullWaves).T)
        plt.plot(realEnvInterp)
        plt.legend(['Exp 1','Exp 2', 'Fit 1', 'Fit 2'])

    waves = np.array(waves)
    
    #print('gamma = ',1/popt[0])
    #print('sigma = ',popt[2])
        
    return exTime, waves, delay, table, vHom, np.array(fullWaves)
