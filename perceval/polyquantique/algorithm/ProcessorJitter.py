
import perceval as pcvl
from perceval.polyquantique.algorithm.DistributionEnvelope import Exponential, Overlap , qr_mgs_decompose, characterize_basis
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import itertools


class ProcessorJitter():



    def __init__(self, bs_jitter,circuit):
        processor = pcvl.Processor("Naive", circuit)
        self.inputs = [pcvl.BasicState(state) for state in bs_jitter.bs_vector]
        Analyse = pcvl.algorithm.Analyzer(processor,self.inputs,"*")
        pcvl.pdisplay(Analyse)
        self.analyse = Analyse.compute()
        self.output_vect,self.output_prob = self.make_results(bs_jitter,processor)
        

    def make_results(self,bs_jitter,processor):
        Analyse = self.analyse
        output_vect = np.zeros((len(Analyse['output_states']),bs_jitter.size_vect_array))
        output_prob = np.zeros((len(Analyse['output_states']),1))
        for output_states in range(len(Analyse['output_states'])):
            for input_states in range(len(Analyse['input_states'])):
                for vect in range(bs_jitter.bs.n):
                    output_vect[output_states,:] += Analyse['results'][input_states,output_states]*bs_jitter.coef_list[input_states,vect]*bs_jitter.new_base[vect,:]
            output_prob[output_states] = sc.integrate.simps(output_vect[output_states,:],bs_jitter.space_array)
        return output_vect,output_prob

    def print_vect(self,bs_jitter):
        plt.plot(bs_jitter.space_array,self.output_vect.T)
        plt.show()

    def print_output(self):
        for output_states in range(len(self.analyse['output_states'])):
            print(self.analyse['output_states'][output_states],self.output_prob[output_states] )
        print(np.sum(self.output_prob))