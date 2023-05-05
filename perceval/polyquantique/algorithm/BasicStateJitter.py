

import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential, Overlap , qr_mgs_decompose, characterize_basis, Gaussian
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import itertools


class BasicStateJitter():
    """Create a Basic with envelopes that can have offset in time
    """

    def __init__(self,state,offset,source=None,methode='Gram_Schmidt'):
        """Create a new orthogonal base for the envelopes and create the 
        associate probabilities to be used in the ProcessorJitter

        :param state: BasicState of a photon
        :type state: str
        :param offset: List of the offset of the photons. Must be the same lenght as state
        :type offset: list, optional
        :param source: The SourceJitter, defaults to None
        :type source: SourceJitter, optional
        :param methode: Methode d'orthogonalisation
        :type methode: str, optional
        """
        
        self.bs = pcvl.BasicState(state)
        if source is None:      
            source = SourceJitter(["Dirac",1],["Exponential",0.5])
        if self.bs.m != np.array(offset).size:  # check if the jitter is the right size
            raise TypeError("State and offset must have the same lenght")
        self.offset = offset
        self.space_array = source.space_array
        self.vector_list = self.Vector_list(source)
        self.coef_matrix, self.vector_list_ortho, self.new_base= self.orthogonalisation(methode=methode)
        self.coef_list ,self.bs_vector =  self.make_states()

    def Vector_list(self,source):
        """Create the list of list that contain all the envelope. 
        Only the one with at least 1 photon

        :param source: The SourceJitter, defaults to None
        :type source: SourceJitter
        :return: list of list of the envelopes
        :rtype: list
        """
        vector_list = np.zeros((self.bs.n,source.space_array.size))
        jitter = self.offset[np.array(list(self.bs))==1]
        for photon in range(self.bs.n):
            vector_list[photon,:] = source.envelope_vector([source.space_array-jitter[photon]])
        return vector_list

    def orthogonalisation(self,methode='Gram_Schmidt'):
        """Orthogonalise vector from vector list

        :param methode: Methode use to orthogonalise, defaults to 'Gram_Schmidt'
        :type methode: str, optional
        :return: coef_matrix,vector_list_ortho,new_base
        :rtype: list,list,list
        """
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
    

    @ property
    def print_vect(self):
        """Plot envelopes and the new base
        """
        plt.plot(self.space_array,self.vector_list.T)
        plt.ylabel('Amplitude normalisée')
        plt.xlabel('Temps (s)')
        plt.title('Enveloppe des photons')
        plt.show()
        plt.plot(self.space_array,self.new_base.T)
        plt.ylabel('Amplitude normalisée')
        plt.xlabel('Temps (ns)')
        plt.title('Enveloppe normalisée des photons')
        plt.show()

    def make_states(self):
        """Create the new states as a list of list that correspond 
        to eache entry with the associate property

        :return: coef_list ,bs_vector
        :rtype: list,list
        """
        idx_matrix = list(itertools.product(np.arange(self.bs.n),repeat=self.bs.n))
        coef_list = np.ones(len(idx_matrix))
        for idx in range(len(idx_matrix)):
            for line in range(len(idx_matrix[idx])):
                coef_list[idx]*= self.coef_matrix[line,idx_matrix[idx][line]]**2
        bs_vector = []
        for idx in range(len(idx_matrix)):
            list_element = []
            for vector in range(self.bs.n) :
                element = list(np.where(np.array(idx_matrix[idx])==vector,1,0))
                if self.bs.n != self.bs.m :
                    idx_zeros = np.where(np.array(list(self.bs)) == 0)[0]
                    for idxz in idx_zeros :
                        element.insert(idxz,0)
                list_element.append(tuple(element))
            bs_vector.append(list_element)
        return coef_list ,bs_vector
    

class SourceJitter():
    """Create a source for all the Jitter objects. It take the enveloppe of the 
    photon and the distribution error of the source in count. It also take the range of the analyse.
    """

    def __init__(self,distribution_list,envelope_list,range = [-10e-12,30e-12],size_vect = 10000):
        f"""Put the first argument as the description and all the over as caracteristique

        :param distribution_list: ['type of distribution', param1,param2] can take Cauchy and Dirac distribution.
        :type distribution_list: list
        :param envelope_list: ['type of envelope', param1,param2] can take Exponential and Gaussian distribution.
        :type envelope_list: list
        :param range: The range of the analyse of the photon in seconds, defaults to [-10e-12,30e-12]
        :type range: list, optional
        :param size_vect: Resolution of the envelope
        :type size_vect: int, optional
        """
        self.distribution=distribution_list[0]
        self.distribution_arg = distribution_list[1:]
        self.envelope = envelope_list[0]
        self.envelope_arg = envelope_list[1:]
        self.range = range
        self.size_vect_array = size_vect
        self.space_array = np.linspace(range[0],range[1],self.size_vect_array)

    def distribution_jitter(self,size):
        """Return a list of offset according to the distribution used

        :param size: Number of photon  
        :type size: int
        :return: List of offset
        :rtype: list
        """
        if self.distribution == "Cauchy":
            return np.random.standard_cauchy(size)
        if self.distribution == "Dirac":
            return np.array([0]*size)

    def envelope_vector(self,x_list):
        """ Give the shape of the envelope with a list of abscissas 

        :param x_list: Abscissas of the envelope
        :type x_list: numpy.ndarray
        :raises TypeError: If the envelope is not implemented
        :return: Envelope of the photon as 1D array
        :rtype: numpy.ndarray
        """
        if self.envelope == "Exponential":
            return Exponential(np.array(x_list),self.envelope_arg[0])
        elif self.envelope == "Gaussian" :
            return Gaussian(np.array(x_list), self.envelope_arg[0])
        else :
            raise TypeError("Unknow envelope")

        

