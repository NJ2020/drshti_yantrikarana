"""
Usage:

About: Script to create nd numpay array of images and labels for large
datasets using pytables

Author: Satish Jasthi
"""

import numpy as np
import tables
from PIL import Image

from drshti_yantrikarana import data_dir, resize_shape, channel_depth


# TODO: Add logger
class CreateNdDataArray(object):

    def __init__(self):
        # create a folder to store nd data array in a HDF5 file
        self.hdf5_file_path = data_dir.joinpath(data_dir.parent, 'HDF5_data_files')
        self.hdf5_file_path.mkdir(parents=True, exist_ok=True)
        self.hdf5_file = self.hdf5_file_path.joinpath(self.hdf5_file_path, 'Data.h5')
        self.hdf5_file = tables.open_file(self.hdf5_file, mode='w')

    def getNdArray(self):
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
        num_images = 0
        labels = []
        # add image data to images_earray
        for image_path in data_dir.glob('*/*'):
            labels.append(image_path.stem)
            img = Image.open(image_path)
            img = img.resize(resize_shape, Image.ANTIALIAS)
            img_ar = np.array(img)[:, :, :3]
            images_earray.append(img_ar[None])
            image_channel_wise_sum += img_ar
            num_images += 1

        # image_channel_wise_mean = np.sum(image_channel_wise_sum, axis=(0, 1)) / num_images
        # TODO: Add code to calculate mean and std for images

        self.hdf5_file.create_array(where=self.hdf5_file.root,
                                    name='labels',
                                    obj=labels
                                    )
        self.hdf5_file.close()

if __name__ == '__main__':
    # TODO: Write test data for CreateDataArray
    o = CreateNdDataArray()
    o.getNdArray()
