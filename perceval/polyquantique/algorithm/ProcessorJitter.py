
import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential, Overlap , qr_mgs_decompose, characterize_basis
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import itertools
import time as time 


class ProcessorJitter():
    """Take a Processor object and a BasicStateJitter object and compute the output of the circuit.
    """


    def __init__(self, bs_jitter,processor):
        """Make directly the result and store the analyse of the circuit in analyse, 
        the output state and their probabilities in output_vect and output_prob

        :param bs_jitter: The BasicStateJitter you want to analyse
        :type bs_jitter: BasicStateJitter
        :param processor: The processor you want to use 
        :type processor: Processor
        """
        inputs = []
        for state in list(itertools.product([0,1],repeat = bs_jitter.bs.n)):
            state0 = list(state)
            if bs_jitter.bs.n != bs_jitter.bs.m :
                idx_zeros = np.where(np.array(list(bs_jitter.bs)) == 0)[0]
                for idxz in idx_zeros :
                    state0.insert(idxz,0)
                inputs.append(pcvl.BasicState(state0))
            else:
                inputs.append(pcvl.BasicState(state0))
        self.inputs = inputs
        self.n = bs_jitter.bs.n
        analyser = pcvl.algorithm.Analyzer(processor,self.inputs,"*")
        self.analyse = analyser.compute()
        self.output_vect,self.output_prob = self.make_results(bs_jitter,self.analyse)
        

    def make_results(self,bs_jitter,Analyse):
        """Compute the output of the system

        :param bs_jitter: The basic state at the entrance of the system,
        :type bs_jitter: BasicStateJitter
        :return: Analyse from the processor 
        :rtype: Dictionnary
        """
        output_vect = [list(Analyse['output_states'][i]) for i in range(len(Analyse['output_states']))]
        list_in = [list(Analyse['input_states'][i]) for i in range(len(Analyse['input_states']))]
        output_prob = np.zeros((len(Analyse['output_states']),1))
        sizestate = len(bs_jitter.bs_vector)
        for state in range(len(bs_jitter.bs_vector)):
            init = time.time()
            print(state,'/',sizestate)
            listv = bs_jitter.bs_vector[state]*1
            print(listv)
            n=listv.count(tuple([ 0 for i in range(bs_jitter.bs.m)]))
            for i in range(n): 
                listv.remove(tuple([ 0 for i in range(bs_jitter.bs.m)]))
            idx_matrix = list(itertools.product(np.arange(len(Analyse['output_states'])),repeat=bs_jitter.bs.n-n))
            print(len(idx_matrix))
            for idx in range(len(idx_matrix)):
                prob = 1
                vec = np.zeros(bs_jitter.bs.m)
                for line in range(len(idx_matrix[idx])):
                    #if list(listv[line]) != [ 0 for i in range(bs_jitter.bs.m)]:
                    prob*= Analyse['results'][list_in.index(list(listv[line])),idx_matrix[idx][line]]
                    vec += output_vect[idx_matrix[idx][line]]
                if prob != 0:
                    output_prob[output_vect.index(list(vec))] += prob*bs_jitter.coef_list[state]
            print('temps =',time.time()-init)
        return output_vect,output_prob

    def set_post_process(self):
        # Not coded yet
        pass

    @ property
    def print_output(self):
        """Print all the states and their associate probabilities
        """
        for output_states in range(len(self.analyse['output_states'])):
            if np.sum(list(self.analyse['output_states'][output_states])) == self.n:
                print(self.analyse['output_states'][output_states],self.output_prob[output_states])
        print('Probabilite totale =',np.sum(self.output_prob))