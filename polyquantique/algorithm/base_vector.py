

import perceval as pcvl
import numpy as np

class base_vector():
    """
    Base vector for granularity algorithm


    Input :

    """

    def __init__(self,state:str,offsets = None,envelope:str = 'gaussian'):
        self.bs = pcvl.BasicState(state)
        self.offsets = offsets
        self.envelope = envelope
    
    def envelope_array(self):
        array=np.array([])
        if self.envelope == "gaussian":
            array = 1
        
        return array

