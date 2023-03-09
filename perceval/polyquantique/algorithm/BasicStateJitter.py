

import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential, Overlap , qr_mgs_decompose, characterize_basis
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import itertools


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
            source = Source(["Dirac",1],["Exponential",0.5])# 2/90e-12
        if offset is None:    #set jitter to 0 if no array is given
            offset = np.zeros(self.bs.m)
        if self.bs.m != np.array(offset).size:  # check if the jitter is the right size
            raise TypeError("State and jitter must have the same dimension")
        self.offset = offset
        self.source = source
        self.size_vect_array = 10000
        self.space_array = np.linspace(-10,30,self.size_vect_array)
        self.vector_list = self.Vector_list(offset,source)
        self.coef_matrix, self.vector_list_ortho, self.new_base= self.orthogonalisation(methode='Gram_Schmidt')
        self.coef_list ,self.bs_vector =  self.make_states()

    def orthogonalisation(self,methode='Gram_Schmidt'):
        if methode=='Gram_Schmidt':
            new_base,n = qr_mgs_decompose(self.vector_list.T)[0],0
            while characterize_basis(new_base) > 2*10**-20 and n<150:
                    new_base = qr_mgs_decompose(new_base)[0]
                    n+=1
            new_base = new_base.T
            for i in range(len(new_base)):
                new_base[i] = new_base[i] / np.sqrt(Overlap(new_base[i], new_base[i], self.space_array))
        coef_matrix=np.zeros((self.bs.n,self.bs.n))
        for i in range(self.bs.n):
            for j in range(i+1):
                coef_matrix[i,j]=sc.integrate.simps(self.vector_list[i] * new_base[j], self.space_array)

        vector_list_ortho=np.zeros(self.vector_list.shape)
        for i in range(self.bs.n):
            for j in range(self.bs.n):
                vector_list_ortho[i]=vector_list_ortho[i]+coef_matrix[i,j]*new_base[j]

        return coef_matrix,vector_list_ortho,new_base
    
    def Vector_list(self,jitter,source):
        vector_list = np.zeros((self.bs.n,self.space_array.size))
        jitter = jitter[np.array(list(self.bs))==1]
        for photon in range(self.bs.n):
            vector_list[photon,:] = source.envelope_vector([self.space_array-jitter[photon]])
        return vector_list

    def print_vect(self):
        plt.plot(self.space_array,self.vector_list.T)
        plt.show()
        plt.plot(self.space_array,self.new_base.T)
        plt.show()

    def make_states(self):
        bs_vector = list(itertools.product([0,1],repeat = self.bs.n))[1:]
        coef_list = np.zeros((len(bs_vector),self.bs.n))
        for state in range(len(bs_vector)):
            for j in range(self.coef_matrix.shape[1]):
                coef_column = 1
                for i in range(self.coef_matrix.shape[0]):
                    if bs_vector[state][i] == 1:
                        coef_column *= self.coef_matrix[i,j]
                    else :
                        coef_column *= np.sum(np.delete(self.coef_matrix[i,:],j))
                coef_list[state,j] = coef_column
        if self.bs.n != self.bs.m :
            idx_zeros = np.where(np.array(list(self.bs)) == 0)[0]
            for idx in idx_zeros :
                for state in range(len(bs_vector)):
                    state0 = list(bs_vector[state])
                    state0.insert(idx,0)
                    bs_vector[state] = tuple(state0)
        return coef_list ,bs_vector
    

class Source():
    """
    Make a source for the granularity methode

    """

    def __init__(self,distribution_list,envelope_list):
        # distribution_list[nom,nom du fichier]
        # nom du fichier - self.envelope arg
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
            # code here experimental envelope were argument self.envelope_arg[0] = "file_name.json"
            pass
        else :
            raise TypeError("Unknow envelope")
        

