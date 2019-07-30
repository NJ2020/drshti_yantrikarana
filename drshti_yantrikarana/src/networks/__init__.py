"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""

class DLNetworks(object):

    def __init__(self, model_type:str=None):
        """

        :param model_type: str, can be 'classification', 'objectDetection' or 'ObjectSegmentation'
        """
        self.model_type = model_type