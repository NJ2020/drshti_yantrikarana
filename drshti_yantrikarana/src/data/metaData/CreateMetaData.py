"""
Usage: python CreateMetaData.py

About: Script to create meta data for image data stored in the format mentioned in README Data Format Section

Author: Satish Jasthi
"""

import logging
import argparse

import pandas as pd
import numpy as np
import tables
from PIL import Image

from drshti_yantrikarana.src.data.storage.database import mysql_engine, mean_std_coltn
from drshti_yantrikarana.config import data_dir, hdf5_data

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()

parser.add_argument('--metaDataLevel',
                    help="'basic' indicates to create basic meta data and 'mean_std' to calculate mean and std of images"
                         "at channel level",
                    default='basic',
                    type=str
                    )

# TODO: Add function to fix the corrupted images

class ImageMetaData(object):

    def __init__(self):
        pass

    @staticmethod
    def create_meta_data() -> None:
        """
        Function to create meta data from data dir
        :return: None
        """
        logging.info('Creating metadata')
        # create meta data at class level
        classes = []
        class_wise_data_samples_count = {}
        raw_data = data_dir/'RawData'
        for class_dir in raw_data.glob('*'):
            classes.append(class_dir.name)
            class_wise_data_samples_count[class_dir.name] = len(list(class_dir.glob('*')))

        # create a table to capture class wise number of images
        class_wise_images = pd.DataFrame(data={'Class': list(class_wise_data_samples_count.keys()),
                                               'Number of images': list(class_wise_data_samples_count.values())})
        class_wise_images.to_sql('Class_wise_image_count', mysql_engine, if_exists='replace')

        # create meta data at individual image level
        image_level_meta_data = pd.DataFrame(columns=['Image', 'height', 'width', 'depth'])
        corrupted_images = pd.DataFrame(columns=['Image'])
        for image in raw_data.glob('*/*'):
            try:
                img = Image.open(image)
                image_level_meta_data = image_level_meta_data.append({'Image': image.as_posix(),
                                                                      'width': img.size[0],
                                                                      'height': img.size[1],
                                                                      'depth': np.array(img).shape[-1]
                                                                      },
                                                                     ignore_index=True)
            except IOError:
                logging.warning(f'Unable to read image {image.as_posix()}')
                corrupted_images = corrupted_images.append({'Image': image.as_posix()}, ignore_index=True)
                continue
        logging.info('Saving meta data to db')

        if corrupted_images.shape[0] > 1:
            corrupted_images.to_sql('Corrupted_images', mysql_engine, if_exists='replace')
        image_level_meta_data.to_sql('Image_level_meta_data', mysql_engine, if_exists='replace')


    def getDataMeanAndStd(self)->(np.array, np.array):
        """
        Method to get data mean and standard deviation for whole dataset
        :return:
        """
        logging.info('Creating mean and std for entire data')

        hdf5_curs = tables.open_file(hdf5_data, mode='r')
        imgs_nd_ar = hdf5_curs.root.images
        index, mean_list, std_list = 0, [],[]
        while index < imgs_nd_ar.shape[0]:
            temp_img_ar = imgs_nd_ar[index: index+1000, :, :, :]
            index = index + 100
            mean_list.append(np.mean(temp_img_ar, axis=(0,1)))
            std_list.append(np.std(temp_img_ar, axis=(0,1)))
        mean, std = np.mean(mean_list), np.mean(std_list)
        # TODO Add support to push and read mean and std from mongodb
        # mean_std_coltn.insert_one({'mean': mean, 'std': std})
        return mean, std

def main():
    obj = ImageMetaData()
    obj.create_meta_data()
    print(obj.getDataMeanAndStd())

if __name__ == '__main__':
    main()
