"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
import logging
from pathlib import Path

from unittest import TestCase, main
from pandas import read_sql
from PIL import Image

from drshti_yantrikarana.src.data.storage.database import mysql_engine
from drshti_yantrikarana.config import data_dir

logging.basicConfig(level=logging.INFO)


class TestImageMetaData(TestCase):

    def test_create_meta_data_method(self):
        Class_wise_image_count = read_sql('Class_wise_image_count', con=mysql_engine)
        Corrupted_images = read_sql('Corrupted_images', mysql_engine)

        # test for number of classes captured in data
        self.assertEqual(Class_wise_image_count.Class.nunique(), len(list(data_dir.glob('*'))))
        
        # test for number of samples captured  per class
        class_samples_count = {class_dir.name:len(list(class_dir.glob('*'))) for class_dir in data_dir.glob('*')}
        for class_name, num_samples in class_samples_count.items():
            self.assertEqual(num_samples, class_samples_count[class_name])
    
        # test for number of corrupted images
        corrupted_images = 0
        image_path: Path()
        for image_path in data_dir.glob('*/*'):
            try:
                image = Image.open(image_path)
            except IOError:
                logging.INFO(f'Unable to read image: {image_path}')
                corrupted_images+=1
                continue
        self.assertEqual(Corrupted_images.Image.nunique(), corrupted_images)

if __name__ == '__main__':
    main()