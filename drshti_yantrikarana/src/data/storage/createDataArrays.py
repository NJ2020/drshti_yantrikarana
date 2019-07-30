"""
Usage:

About: Script to create nd numpay array of images and labels for large
datasets using pytables

Author: Satish Jasthi
"""
import logging

import numpy as np
import tables
from PIL import Image
import tensorflow as tf

from drshti_yantrikarana import data_dir, resize_shape, channel_depth
from drshti_yantrikarana.src.data.preprocessing.resizeImage import resize_image

logging.basicConfig(level=logging.DEBUG)

class CreateNdDataArray(object):

    def __init__(self, mode=None):
        # create a folder to store nd data array in a HDF5 file
        self.mode=mode
        self.hdf5_file_path = data_dir.joinpath(data_dir.parent, 'HDF5_data_files')
        self.hdf5_file_path.mkdir(parents=True, exist_ok=True)
        if self.mode=='train':
            self.hdf5_file = self.hdf5_file_path.joinpath(self.hdf5_file_path, 'TrainData.h5')
        elif self.mode=='test':
            self.hdf5_file = self.hdf5_file_path.joinpath(self.hdf5_file_path, 'TestData.h5')
        self.hdf5_file = tables.open_file(self.hdf5_file, mode='w')

    def createNdArray(self):
        """
        Method to create an Nd array to store images and an array to store respective labels
        :return: None
        """

        # create expandable array to add image data to numpy array in loop
        images_earray = self.hdf5_file.create_earray(where=self.hdf5_file.root,
                                                     name='images',
                                                     atom=tables.UInt8Atom(),
                                                     shape=[0, resize_shape[0], resize_shape[1], channel_depth])

        image_channel_wise_sum = np.zeros(shape=[resize_shape[0], resize_shape[1], channel_depth])
        num_images, labels = 0, []

        # add image data to images_earray
        raw_data = data_dir/"RawData"
        logging.info('Creating HDF5 numpy data from raw images..........................................................')
        for image_path in raw_data.glob('*/*'):
            labels.append(image_path.stem)
            img = Image.open(image_path)

            # resize image using predifined dimensions in config
            img_tensor = tf.convert_to_tensor(np.array(img))
            img_rz = resize_image(img_tensor).numpy()
            img_ar = np.array(img_rz)[:, :, :3]
            images_earray.append(img_ar[None])
            image_channel_wise_sum += img_ar
            num_images += 1

        self.hdf5_file.create_array(where=self.hdf5_file.root,
                                    name='labels',
                                    obj=labels
                                    )
        self.hdf5_file.close()

if __name__ == '__main__':
    # TODO: Write test data for CreateDataArray
    o = CreateNdDataArray(mode='train')
    o.createNdArray()
