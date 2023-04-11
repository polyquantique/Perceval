import perceval as pcvl
import numpy as np
import matplotlib.pyplot as plt
import itertools
from Functions import *
import matplotlib.pyplot as plt
import scipy as sc
from scipy import signal
from tabulate import tabulate
from Convolution import createWaves


def create_CRand(i,coef):
    CRand = []
    RawInputs = []
    Expected = []
    if i==0:
        CRand.append(np.array([[1,0],[0,0],[np.sqrt(1-coef ** 2),coef],[0,0]])) #Input = [1,0,1,0] = 0,0 exp = 0,0
        RawInputs.append([1,0,1,0])
        Expected = [1,0,1,0]
    elif i==1:
        CRand.append(np.array([[1,0],[0,0],[0,0],[np.sqrt(1-coef ** 2),coef]])) #Input = [1,0,0,1] = 0,1 exp = 0,1
        RawInputs.append([1,0,0,1])
        Expected = [1,0,0,1]
    elif i==2:
        CRand.append(np.array([[0,0],[1,0],[np.sqrt(1-coef ** 2),coef],[0,0]])) #Input = [0,1,1,0] = 1,0 exp = 1,1
        RawInputs.append([0,1,1,0])
        Expected = [0,1,0,1]
    elif i==3:
        CRand.append(np.array([[0,0],[1,0],[0,0],[np.sqrt(1-coef ** 2),coef]])) #Input = [0,1,0,1] = 1,1 exp = 1,0
        RawInputs.append([0,1,0,1])
        Expected = [0,1,1,0]
    else :
        print('Epic Fail')
        return
    return CRand, np.squeeze(RawInputs), Expected

# Create the C manually because it is not yet implemented
statesdict = {
    pcvl.BasicState([1,0,1,0]) : "00",
    pcvl.BasicState([1,0,0,1]) : "01",
    pcvl.BasicState([0,1,1,0]) : "10",
    pcvl.BasicState([0,1,0,1]) : "11"
}

def create_inputs(enterFunc,Coefs,Dictionnary = None): 
    #enterFunc is the distribution in space of the entry in the system ex: [1,1]
    if any(np.array(enterFunc)>1):
        raise Exception('Cannot be more than 1 photon per spatial mode')
        
    internMode = np.sum(enterFunc)
    spatialMode = len(enterFunc)
    nbPhoton = np.sum(enterFunc)
    L = spatialMode*internMode 

    Inputs = []
    for x in itertools.combinations(range(L), nbPhoton) :
        inputsTemps = [1 if i in x else 0 for i in range(L)]
        Inputs.append(np.array(inputsTemps))
    
    realInputs = []

    for i in range(len(Inputs)):
        compare = np.zeros(spatialMode)
        for ii in range(0,L,spatialMode):
            compare += Inputs[i][ii:ii+spatialMode]

        if np.all(compare == enterFunc):
            realInputs.append(Inputs[i])
    #print(realInputs)
    c = []
    Arr = np.squeeze(Coefs)
    
    for i in range(len(realInputs)):
        ArrCond = np.array(realInputs[i]).reshape(internMode,spatialMode).T
        y = np.ma.masked_array(Arr, abs(ArrCond-1))
        c.append(np.prod(y))

    k = 0

    if isinstance(Dictionnary,dict) and len(Dictionnary)!=0:
        testdict = {}

        iterkeys = itertools.permutations(Dictionnary.keys(),2)
        iterval = itertools.permutations(Dictionnary.values(),2)
        for i,j in zip(iterkeys,iterval) :
            dictkey = '|' + i[0] + ',' + i[1] + '>'
            dictval = j[0] + j[1]
            testdict[dictkey] = pcvl.BasicState(dictval)
        for i in realInputs:
            nameOfState = str(pcvl.BasicState(i))
            if k == 0:
                InputsBS = testdict[nameOfState]
                
            else:
                InputsBS = InputsBS + testdict[nameOfState]

            k += 1
        #Add the coefficient in the state vector
        k = 0
        for state,amplitude in InputsBS.items():

            InputsBS[state] = amplitude*c[k]
            k+=1
        
        return InputsBS,testdict
    else:     
        for i in realInputs:
            if k == 0 :
                InputsBS = pcvl.BasicState(i)
            else :
                InputsBS = InputsBS + pcvl.BasicState(i)
            k += 1
        #Add the coefficient in the state vector
        k = 0
        for state,amplitude in InputsBS.items():

            InputsBS[state] = amplitude*c[k]
            k+=1
        
        return InputsBS

def calculateCNOT(p):
    names = ['./acquired_data/xp_00/data.json','./acquired_data/xp_01/data.json','./acquired_data/xp_02/data.json','./acquired_data/xp_03/data.json','./acquired_data/xp_04/data.json','./acquired_data/xp_05/data.json','./acquired_data/xp_06/data.json','./acquired_data/xp_07/data.json','./acquired_data/xp_08/data.json','./acquired_data/xp_09/data.json']
    compareVal = []
    realValMax = []
    realValMin = []
    delayVal = []
    for i in names:
        time, waves, delay, table, vHom, fullWaves = createWaves(i,doYouPlot = False,timeArray=2 ** 14 +1 )
        new_base,coeffsMGS = modified_Schmidt(waves,time)
        coeff = coeffsMGS[1,1]
        TableValue = []


        statesProb = dict.fromkeys(statesdict.keys(),0)

        [C,Inputs,Expected] = create_CRand(2,coeff)
        C = np.array(C)
        C = C*0.95
        InputsBS = create_inputs(Inputs,C)

        print(InputsBS)


        pcvl.pdisplay(p, recursive = True)
        realOutput = {}
        for i in range(len(InputsBS)):
            miniState = InputsBS[i]
            #print(miniState,':',InputsBS[miniState])
            p.with_input(miniState)
            output = p.probs()['results']
            #print(output)
            for ii in output.keys():
                if ii in realOutput.keys():
                    realOutput[ii] = realOutput[ii] + output[ii] * abs(InputsBS[InputsBS[i]]) ** 2
                else:
                    realOutput[ii] = output[ii] * abs(InputsBS[InputsBS[i]]) ** 2

        Prob = 0
        
        for states,val in realOutput.items():
            ls = np.array(states)
            Cond = ls[0:4] + ls[4:]
            tempState = pcvl.BasicState(Cond)
            #print(tempState,val)
            if tempState in statesProb.keys():
                statesProb[tempState] += val
        
        for i,j in statesProb.items():
            print(i,j)
        results = {key: value / 4 for key, value in statesProb.items()}
        #print(results)

        TableValue.append(results)
        A = pcvl.BasicState([0,1,0,1])
        compareVal.append(statesProb[A])
        realValMax.append(table[2,3])
        realValMin.append(table[3,2])
        delayVal.append(delay)

    return TableValue,compareVal,realValMax,realValMin,delayVal