"""
About: Script to create TF records for image data by reading data
stored from a hdf5 file

Author: Satish Jasthi
"""
import numpy as np
import tables
import tensorflow as tf

from drshti_yantrikarana import hdf5_data, tfRecord_data


class TfRecords(object):

    def __init__(self):
        self.hdf5_data = tables.open_file(hdf5_data, mode='r')
        tfRecord_data.parent.mkdir(parents=True, exist_ok=True)
        pass

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
        # read images and labels from hdf5
        hdf5_data_arrays = tables.open_file(hdf5_data, w='r')

        # image generator
        image_gen = hdf5_data_arrays.root.images

        # create label index map
        label_names = hdf5_data_arrays.root.labels[:]
        unique_labels = set(label_names)
        label_index_map = {label: index for index, label in enumerate(unique_labels)}

        with tf.compat.v1.python_io.TFRecordWriter(tfRecord_data.as_posix()) as writer:
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
        raw_image_dataset = tf.data.TFRecordDataset(tfRecord_data)
        parsed_image_dataset = raw_image_dataset.map(self._parse_image_function)
        return parsed_image_dataset

if __name__ == '__main__':
    TfRecords().writeTfRecord()