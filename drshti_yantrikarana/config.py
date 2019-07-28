"""
About: Configuration file for whole project

Author: Satish Jasthi
"""

# database configs
from pathlib import Path

db_username = 'root'
db_password = 'password'
db_name = 'drshit_yantrikarana'

# data configs##########################################################################################################
data_dir = Path('/Users/satishjasthi/Documents/Professional/ML/drshti_yantrikarana/data')
hdf5_data = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/Data.h5')
tfRecord_data = data_dir.joinpath(data_dir.parent, 'TFRecords_data_files/Data.tfrecords')


# data preprocessing####################################################################################################
resize_shape = (224,224)
# mu and sigma must be tf.float64
# TODO add code to calculate data_mu and data_sigma
data_mu = ''
data_sigma = ''
channel_depth = 3

# data augmentation#####################################################################################################
# for random rotation
rotation_min = 10
rotation_max = 45