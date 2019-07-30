"""
Reference:
    1. to convert dataset object to images and labels tensors
        Speeding up Keras with tfrecord datasets(https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-
        datasets-5464f9836c36)
Usage:

About: Script to create Train and Evaluaiton data generator from Tfrecords

Author: Satish Jasthi
"""
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from drshti_yantrikarana.config import (datagen_featurewise_center, datagen_samplewise_center,
                                        datagen_featurewise_std_normalization, datagen_samplewise_std_normalization,
                                        datagen_zca_whitening, datagen_zca_epsilon, datagen_rotation_range,
                                        datagen_width_shift_range, datagen_height_shift_range, datagen_brightness_range,
                                        datagen_shear_range, datagen_zoom_range, datagen_channel_shift_range,
                                        datagen_fill_mode,
                                        datagen_cval, datagen_horizontal_flip, datagen_vertical_flip, datagen_rescale,
                                        datagen_preprocessing_function, datagen_data_format, datagen_validation_split,
                                        datagen_datagen_dtype, data_mu, data_sigma, resize_shape,
                                        datagen_SHUFFLE_BUFFER, datagen_BATCH_SIZE)
from drshti_yantrikarana.src.data.storage.createTFRecords import TfRecords
from drshti_yantrikarana.src.data.storage.database import num_classes


def getImageLablesFromDataset(mode=None):
    """
    Function to get image and label tensors from Dataset object
    :return: (images, labels)
    """
    # create data set by reading TfRecords
    dataset = TfRecords(mode=mode).readTfRecord()

    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(datagen_SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(datagen_BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    images = tf.reshape(image, [-1, resize_shape[0], resize_shape[1], 1])

    # Create a one hot array for your labels
    labels = tf.one_hot(label, num_classes)
    return images, labels

datagen_images, datagen_labels = getImageLablesFromDataset()

def getDataGen(datagenMode: str = None, dataType=None) -> ImageDataGenerator:
    """
    Function to create data generator for training
    :param datagenMode: str, 'train' or 'test'
    :param dataType: str can be one of these standard datasets
             - cifar10
             - cifar100
             - mnist
             - fashion_mnist
             or 'custom'
    :return: ImageDataGenerator
    """

    datagen = ImageDataGenerator(ImageDataGenerator(featurewise_center=datagen_featurewise_center,
                                                    samplewise_center=datagen_samplewise_center,
                                                    featurewise_std_normalization=datagen_featurewise_std_normalization,
                                                    samplewise_std_normalization=datagen_samplewise_std_normalization,
                                                    zca_whitening=datagen_zca_whitening,
                                                    zca_epsilon=datagen_zca_epsilon,
                                                    rotation_range=datagen_rotation_range,
                                                    width_shift_range=datagen_width_shift_range,
                                                    height_shift_range=datagen_height_shift_range,
                                                    brightness_range=datagen_brightness_range,
                                                    shear_range=datagen_shear_range,
                                                    zoom_range=datagen_zoom_range,
                                                    channel_shift_range=datagen_channel_shift_range,
                                                    fill_mode=datagen_fill_mode,
                                                    cval=datagen_cval,
                                                    horizontal_flip=datagen_horizontal_flip,
                                                    vertical_flip=datagen_vertical_flip,
                                                    rescale=datagen_rescale,
                                                    preprocessing_function=datagen_preprocessing_function,
                                                    data_format=datagen_data_format,
                                                    validation_split=datagen_validation_split,
                                                    dtype=datagen_datagen_dtype))
    if not dataType=='custom':
        (x_train, y_train), _ = getattr(keras.datasets, dataType).load_data()
        datagen.fit(x_train, y_train)
        del x_train, y_train
    elif dataType=='standard':
        datagen.mean = data_mu
        datagen.std = data_sigma

    return datagen
