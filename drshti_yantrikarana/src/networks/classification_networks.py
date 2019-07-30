"""
Reference: keras.applications
Usage:

About: Script to create popular pretrained models using keras

Author: Satish Jasthi
"""
from tensorflow.python import keras

from drshti_yantrikarana.src.networks import DLNetworks


class ClassificationNetwork(DLNetworks):

    def __init__(self, stand_custom_data:str=None):
        """

        :param stand_custom_data:
        :param stand_custom_data: str can be one of these standard datasets
             - cifar10
             - cifar100
             - mnist
             - fashion_mnist
             or 'custom'
        """
        super().__init__()
        self.model_type = 'classification'
        self.data_type = stand_custom_data
        if not self.data_type in ['cifar10',
                                  'cifar100',
                                  'mnist',
                                  'fashion_mnist',
                                  'custom'
                                  ]:
            raise  Exception(f'Unable to find stand_custom_data out of {self.data_type}')


    def getkerasModel(self,
                   name:str=None,
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000
                   ):
        """
        Method to fetch model based on model name
        :param name: str
        :return:
        Models supported:
            Xception
            VGG16
            VGG19
            ResNet, ResNetV2, ResNeXt
            InceptionV3
            InceptionResNetV2
            MobileNet
            MobileNetV2
            DenseNet
            NASNet
        """
        return getattr(keras.applications, name)(include_top=include_top,
                                                  weights=weights,
                                                  input_tensor=input_tensor,
                                                  input_shape=input_shape,
                                                  pooling=pooling,
                                                  classes=classes)
