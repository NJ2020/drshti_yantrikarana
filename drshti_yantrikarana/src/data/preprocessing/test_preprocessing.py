"""
Reference: 
Usage:

About:

Author: Satish Jasthi
"""
from unittest import TestCase, main

import tensorflow as tf

from drshti_yantrikarana import data_dir, resize_shape
from drshti_yantrikarana.src.data.preprocessing.resizeImage import resize_image


class TestResize_image(TestCase):

    def test_resize_image(self):
        # create folder and add a test image under data/samples dir to test
        resized_image = resize_image(image_path=(data_dir/'samples/test1.jpeg').as_posix())
        self.assertEqual(resize_shape, resized_image.shape[:2])


if __name__ == '__main__':
    main()
