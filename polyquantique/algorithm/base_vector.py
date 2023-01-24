

import perceval as pcvl
import numpy as np

class base_vector():
    """
    Base vector for granularity algorithm


    Input :

    """

    def __init__(self,state:str,jitter = None,source=None):
        self.bs = pcvl.BasicState(state)
        self.jitter = jitter
        self.source = source

    def envelope_array(self):
        array=np.array([])
        if self.envelope == "gaussian":
            array = 1
        return array

    def set_jitter(offset=0,rand=0):
        pass



class source():
    """envelope 
    gamma"""
    def set_envelope():
        pass