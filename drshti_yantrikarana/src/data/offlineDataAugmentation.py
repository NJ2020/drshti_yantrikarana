"""
Reference: 
Usage:

About: To create augmented data for numpy arrays using CPU before training

Author: Satish Jasthi
"""
import logging
import time

import numpy as np
import tables
from PIL import Image
from tensorflow.python import keras

from drshti_yantrikarana.config import (OfflineDataAugImages, OfflineDataAugLabels,
                                        resize_shape, channel_depth, num_classes)
from drshti_yantrikarana.src.data.augmentation.saptialTrasformations import pad_image, random_crop, random_flip, \
    equalize, autocontrast, color, brightness


logging.basicConfig(level=logging.DEBUG)

def createAugmentedData(x_train:np.array=None,
                        y_train:np.array=None)->None:
    """
    Function to create and save augmented data on training images numpy array using following data augmentations
    - padding(4,4) and random crop (32,32)
    - horizontal flip
    - equalize
    - autoContrast
    - color
    - brightness

    """
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    augmetedImageData = tables.open_file(OfflineDataAugImages, mode='w')
    augmetedLabelData = tables.open_file(OfflineDataAugLabels, mode='w')
    
    images_earray = augmetedImageData.create_earray(where=augmetedImageData.root,
                                                 name='images',
                                                 atom=tables.UInt8Atom(),
                                                 shape=[0, resize_shape[0], resize_shape[1], channel_depth])
    labels_earray = augmetedLabelData.create_earray(where=augmetedLabelData.root,
                                                 name='labels',
                                                 atom=tables.UInt8Atom(),
                                                 shape=[0, y_train.shape[1]])



    for image, label in zip(x_train, y_train):

        image = Image.fromarray(image.astype('uint8'))

        # original
        images_earray.append(np.array(image)[None])
        labels_earray.append(label[None])

        # padding
        images_earray.append(np.array(pad_image(image=image,
                                 padding=[[4, 4], [4, 4], [0, 0]]))[None])
        labels_earray.append(label[None])

        # cropping
        images_earray.append(np.array(random_crop(image=image,
                                 height=32,
                                 width=32,
                                 depth=3))[None])
        labels_earray.append(label[None])

        # horizontal flip
        images_earray.append(np.array(random_flip(image=image,
                                flip_mode='h'))[0])
        labels_earray.append(label[None])

        # equalize
        images_earray.append(np.array(equalize(image=image))[0])
        labels_earray.append(label[None])

        # autocontrast
        images_earray.append(np.array(autocontrast(image=image))[0])
        labels_earray.append(label[None])

        # color
        images_earray.append(np.array(color(image=image))[0])
        labels_earray.append(label[None])

        # brightness
        images_earray.append(np.array(brightness(image=image))[0])
        labels_earray.append(label[None])

    augmetedImageData.close()
    augmetedLabelData.close()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    createAugmentedData(x_train, y_train)