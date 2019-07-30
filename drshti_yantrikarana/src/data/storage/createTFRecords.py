"""
About: Script to create TF records for image data by reading data
stored from a hdf5 file

Author: Satish Jasthi
"""
import logging
from pathlib import Path

import numpy as np
import tables
import tensorflow as tf

from drshti_yantrikarana import TrainHdf5_data, TrainTfRecord_data
from drshti_yantrikarana.config import external_data_dir, LOGGER_LEVEL, TestHdf5_data, TestTfRecord_data
from drshti_yantrikarana.src.data.storage.createDataArrays import CreateNdDataArray

logging.basicConfig(level=LOGGER_LEVEL)


class TfRecords(object):

    def __init__(self, mode=None):
        self.mode = mode

        if mode == 'train':
            if TrainHdf5_data.exists():
                self.hdf5_data = tables.open_file(TrainHdf5_data, mode='r')
                TrainTfRecord_data.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise IOError(
                    f'Unable to find {TrainHdf5_data.as_posix()}, Please create hdf5 file before running tf records')
        elif mode == 'test':
            if TestHdf5_data.exists():
                self.hdf5_data = tables.open_file(TestHdf5_data, mode='r')
                TestTfRecord_data.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise IOError(
                    f'Unable to find {TestHdf5_data.as_posix()}, Please create hdf5 file before running tf records')

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert2FeatureMessage(self, image_array: np.array, label: int):
        """
        Method to convert image array to serialized tf.train.Feature message
        :param image_array: np.array
        :param label: int
        :return:
        """
        image_string = tf.io.serialize_tensor(tf.convert_to_tensor(image_array, dtype=tf.uint8))
        image_shape = image_array.shape

        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'depth': self._int64_feature(image_shape[2]),
            'label': self._int64_feature(label),
            'image_raw': self._bytes_feature(image_string.numpy()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def writeTfRecord(self):
        """
        Method to create TFRecord for data set
        :return:
        """
        logging
        # read images and labels from hdf5
        if self.mode == 'train':
            hdf5_data_arrays = tables.open_file(TrainHdf5_data, w='r')
        elif self.mode == 'test':
            hdf5_data_arrays = tables.open_file(TestHdf5_data, w='r')

        # image generator
        image_gen = hdf5_data_arrays.root.images

        # create label index map
        label_names = hdf5_data_arrays.root.labels[:]
        unique_labels = set(label_names)
        label_index_map = {label: index for index, label in enumerate(unique_labels)}

        if self.mode == 'train':
            with tf.compat.v1.python_io.TFRecordWriter(TrainTfRecord_data.as_posix()) as writer:
                for image_arr, label in zip(image_gen, label_names):
                    tf_example = self.convert2FeatureMessage(image_arr, label_index_map[label])
                    writer.write(tf_example.SerializeToString())

            hdf5_data_arrays.close()
        elif self.mode == 'test':
            with tf.compat.v1.python_io.TFRecordWriter(TestTfRecord_data.as_posix()) as writer:
                for image_arr, label in zip(image_gen, label_names):
                    tf_example = self.convert2FeatureMessage(image_arr, label_index_map[label])
                    writer.write(tf_example.SerializeToString())

            hdf5_data_arrays.close()

    @staticmethod
    def _parse_image_function(example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    def readTfRecord(self):
        """
        Method to read tf record format image data
        :return:
        """
        if self.mode == 'train':
            raw_image_dataset = tf.data.TFRecordDataset(TrainTfRecord_data)
        elif self.mode == 'test':
            raw_image_dataset = tf.data.TFRecordDataset(TestTfRecord_data)
        parsed_image_dataset = raw_image_dataset.map(self._parse_image_function)
        return parsed_image_dataset


def main():
    CreateNdDataArray(mode='train').createNdArray()
    tfRecord_creator = TfRecords(mode='train')
    tfRecord_creator.writeTfRecord()


if __name__ == '__main__':
    main()
