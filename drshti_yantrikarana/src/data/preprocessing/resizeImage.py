"""
Reference: 
Usage:

About: Module to resize images to defined resolution in config file


Author: Satish Jasthi
"""
import sys
from pathlib import Path

from PIL import Image
import tensorflow as tf

sys.path.append(Path(__file__).resolve().parent.parent.parent.parent.parent.as_posix())

from drshti_yantrikarana import resize_shape


def resize_image(image_path: str) -> tf.image:
    """
    Function to read an image, convert it into a square image and 
    resize it to standard resolution as defined in config
    :param image_path:str
    :return: tf.image
    """

    # read image
    image_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_image(image_raw)

    # create a central crop wrt larger side to create square image
    h, w = tf.shape(img_tensor)[:2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=(h - w) // 2,
                                                      offset_width=0,
                                                      target_height=w,
                                                      target_width=w
                                                      )
    else:
        cropped_image = tf.image.crop_to_bounding_box(img_tensor,
                                                      offset_height=0,
                                                      offset_width=(w - h) // 2,
                                                      target_height=h,
                                                      target_width=h
                                                      )

    # resize image to predifined res in config
    resized_image = tf.image.resize(cropped_image, size=resize_shape)
    return resized_image
