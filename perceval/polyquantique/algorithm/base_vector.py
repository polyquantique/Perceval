

import perceval as pcvl
import numpy as np
import matplotlib.pyplot as plt

@staticmethod
def expo(gamma,z):
    return (gamma) ** (1 / 2) * np.exp(-gamma * z / 2) * np.heaviside(z, 1)

class base_vector():
    """
    Base vector for granularity algorithm
    """

    def __init__(self,state:str,jitter = None,source=None):
        """
        Input :
        state : str             string for the basic state
        jitter : array          array with the same size of the number of state mode
        source : Source object  

        Properties :
        """
        self.bs = pcvl.BasicState(state)  # set the basic state
        if source is None:      # if no source is given create an exponential one
            source = Source("laurenzien","expo",[1])
        if jitter is None:    #set jitter to 0 if no array is given
            jitter = np.zeros(self.bs.m)
        if self.bs.m != np.array(jitter).size:  # check if the jitter is the right size
            raise TypeError("State and jitter must have the same dimension")
        self.jitter = jitter
        self.source = source
        self.vector_list = self.Vector_list(jitter,source)

    def set_state(self,state):
        pass

    def set_jitter(self,offset=0,rand=0):
        self.jitter = np.ones(self.bs.m)*offset+ np.ones(self.bs.m)*rand*np.random.rand()
        self.vector_list = self.Vector_list(self.jitter,self.source)

    def Vector_list(self,jitter,source):
        vector_list = np.zeros((self.bs.n,100))
        z=np.linspace(-13,13,100)
        for photon in range(self.bs.n):
            vector_list[photon,:] = source.envelope_vector([z-jitter[photon]])
        return vector_list

    def print_vect(self):
        plt.plot(self.vector_list.T)
        plt.show()


class Source():
    """
    Make a source for the granularity methode

    """

    def __init__(self,distribution,envelope,arg):
        self.distribution=distribution
        self.envelope = envelope
        self.arg = arg
        
    def envelope_vector(self,karg):
        if self.envelope == "expo":
            return expo(self.arg[0],karg[0])

