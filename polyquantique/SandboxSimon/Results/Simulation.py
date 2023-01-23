import perceval as pcvl
import numpy as np
import itertools

# Calculate the probability of an outcome for detector
def calculate_proba(n,m,measure,allstateprobs_iterator):
    r"""
    gives the probability of an output measurement
    :param n: number of spatial states
    :param m: number of internal states
    :param measure: array of how much photon we expect for each captor (array 1,n)
    :param allstateprobs_iterator: An iterator containing states and probabilities of the simulation
    :return: float probability
    """
    Proba = 0
    k = 0
    
    for state, probabilitie in allstateprobs_iterator:
        ls = list(state)
        #print(state,probabilitie)
        capt = []
        for i in range(n):
            capt.append(np.sum(ls[i::n]))
        #print(capt)
        if np.all(capt==measure):
            #print(state)
            #print(probabilitie)
            Proba += probabilitie
        k+=1
    return Proba

def create_inputs(n,m,nbPhoton,C1,C2): 
    
    L = nbPhoton*m 

    Inputs = []
    for x in itertools.combinations(range(L), nbPhoton) :
        Inputs.append([1 if i in x else 0 for i in range(L)])

    realInputs = []
    for i in range(len(Inputs)):
        a=0
        for ii in range(n):
            if np.sum(Inputs[i][ii::n]) <= 1:
                a += 1
        if a == 2:
            realInputs.append(Inputs[i])

    c = []
    Arr = np.squeeze(np.array([[C1],[C2]]))
    n = np.size(Arr,0)
    m = np.size(Arr,1)
    for i in range(len(realInputs)):
        ArrCond = np.array(realInputs[i]).reshape(m,n).T
        y = np.ma.masked_array(Arr, abs(ArrCond-1))
        c.append(np.prod(y))
    k = 0
    for i in realInputs:
        if k == 0 :
            InputsBS = pcvl.BasicState(i)
        else :
            InputsBS = InputsBS + pcvl.BasicState(i)
        k+=1
    #Add the coefficient in the state vector
    k = 0
    for state,amplitude in InputsBS.items():

        InputsBS[state] = amplitude*c[k]
        k+=1
    return InputsBS