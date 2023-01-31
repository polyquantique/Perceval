

import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential
import numpy as np
import matplotlib.pyplot as plt

class BasicStateJitter():
    """
    Base vector for granularity algorithm
    """

    def __init__(self,state:str,offset = None,source=None):
        """
        Input :
        state : str             string for the basic state
        jitter : array          array with the same size of the number of state mode
        source : Source object  

        Properties :
        """
        self.bs = pcvl.BasicState(state)  # set the basic state
        if source is None:      # if no source is given create an exponential one
            source = Source(["Dirac",1],["Exponential",1])# 2/90e-12
        if offset is None:    #set jitter to 0 if no array is given
            offset = np.zeros(self.bs.m)
        if self.bs.m != np.array(offset).size:  # check if the jitter is the right size
            raise TypeError("State and jitter must have the same dimension")
        self.offset = offset
        self.source = source
        self.vector_list = self.Vector_list(offset,source)

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

    def __init__(self,distribution_list,envelope_list):
        self.distribution=distribution_list[0]
        self.distribution_arg = distribution_list[1:]
        self.envelope =envelope_list[0]
        self.envelope_arg = envelope_list[1:]

    def distribution_jitter(self):
        if self.distribution == "Cauchy":
            return np.random.standard_cauchy()
        if self.distribution == "Dirac":
            return 0

    def envelope_vector(self,x_list):
        if self.envelope == "Exponential":
            return Exponential(np.array(x_list),self.envelope_arg[0])
        if self.envelope == "Experimental":
            # code here experimental envelope were argument self.envelope_arg[0] = "file_name.csv"
            pass
        else :
            raise TypeError("Unknow envelope")
        

