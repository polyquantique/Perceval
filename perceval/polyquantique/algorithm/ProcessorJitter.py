
import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential, Overlap , qr_mgs_decompose, characterize_basis
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import itertools


class ProcessorJitter():



    def __init__(self, bs_jitter,processor):
        processor = processor
        self.inputs = []
        for state in list(itertools.product([0,1],repeat = bs_jitter.bs.n)):
            state0 = list(state)
            if bs_jitter.bs.n != bs_jitter.bs.m :
                idx_zeros = np.where(np.array(list(bs_jitter.bs)) == 0)[0]
                for idxz in idx_zeros :
                    state0.insert(idxz,0)
                self.inputs.append(pcvl.BasicState(state0))
            else:
                self.inputs.append(pcvl.BasicState(state0))
        self.n = bs_jitter.bs.n
        Analyse = pcvl.algorithm.Analyzer(processor,self.inputs,"*")
        pcvl.pdisplay(Analyse)
        self.analyse = Analyse.compute()
        self.output_vect,self.output_prob = self.make_results(bs_jitter,processor)
        

    def make_results(self,bs_jitter,processor):
        Analyse = self.analyse
        output_vect = [list(Analyse['output_states'][i]) for i in range(len(Analyse['output_states']))]
        list_in = [list(Analyse['input_states'][i]) for i in range(len(Analyse['input_states']))]
        output_prob = np.zeros((len(Analyse['output_states']),1))
        for state in range(len(bs_jitter.bs_vector)):
            n=0
            listv = bs_jitter.bs_vector[state]
            for sta in bs_jitter.bs_vector[state]:
                if list(sta) == [ 0 for i in range(bs_jitter.bs.m)]:
                    listv.remove(sta)
                    n+=1
            idx_matrix = list(itertools.product(np.arange(len(Analyse['output_states'])),repeat=bs_jitter.bs.n-n))
            for idx in range(len(idx_matrix)):
                prob = 1
                vec = np.zeros(bs_jitter.bs.m)
                for line in range(len(idx_matrix[idx])):
                    if list(listv[line]) != [ 0 for i in range(bs_jitter.bs.m)]:
                        prob*= Analyse['results'][list_in.index(list(listv[line])),idx_matrix[idx][line]]
                        vec += output_vect[idx_matrix[idx][line]]
                if prob != 0:
                    output_prob[output_vect.index(list(vec))] += prob*bs_jitter.coef_list[state]
        return output_vect,output_prob

    def print_vect(self,bs_jitter):
        plt.plot(bs_jitter.space_array,self.output_vect.T)
        plt.show()

    def print_output(self):
        for output_states in range(len(self.analyse['output_states'])):
            if np.sum(list(self.analyse['output_states'][output_states])) == self.n:
                print(self.analyse['output_states'][output_states],self.output_prob[output_states] )
        print('Probabilite totale =',np.sum(self.output_prob))