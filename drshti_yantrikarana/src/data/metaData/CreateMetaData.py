"""
Usage: python CreateMetaData.py

About: Script to create meta data for image data stored in the format mentioned in README Data Format Section

Author: Satish Jasthi
"""

import logging

import pandas as pd
import numpy as np
from PIL import Image

from drshti_yantrikarana.src.data.storage.database import mysql_engine
from drshti_yantrikarana.config import data_dir

logging.basicConfig(level=logging.INFO)


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

        # create meta data at class level
        classes = []
        class_wise_data_samples_count = {}
        for class_dir in data_dir.glob('*'):
            classes.append(class_dir.name)
            class_wise_data_samples_count[class_dir.name] = len(list(class_dir.glob('*')))

        # create a table to capture class wise number of images
        class_wise_images = pd.DataFrame(data={'Class': class_wise_data_samples_count.keys(),
                                               'Number of images': class_wise_data_samples_count.values()})
        class_wise_images.to_sql('Class_wise_image_count', mysql_engine, if_exists='replace')

        # create meta data at individual image level
        image_level_meta_data = pd.DataFrame(columns=['Image', 'height', 'width', 'depth'])
        corrupted_images = pd.DataFrame(columns=['Image'])
        for image in data_dir.glob('*'):
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
        corrupted_images.to_sql('Corrupted_images', mysql_engine, if_exists='replace')
        image_level_meta_data.to_sql('Image_level_meta_data', mysql_engine, if_exists='replace')


if __name__ == '__main__':
    ImageMetaData.create_meta_data()
