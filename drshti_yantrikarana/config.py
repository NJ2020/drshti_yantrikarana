"""
About: Configuration file for whole project

Author: Satish Jasthi
"""

# database configs
from pathlib import Path

db_username = 'root'
db_password = 'password'
db_name = 'drshit_yantrikarana'

# data configs
data_dir = Path('/Users/satishjasthi/Documents/Professional/ML/drshti_yantrikarana/data')
hdf5_data = data_dir.joinpath(data_dir.parent, 'HDF5_data_files/Data.h5')
tfRecord_data = data_dir.joinpath(data_dir.parent, 'TFRecords_data_files/Data.tfrecords')


# data preprocessing
resize_shape = (224,224)
channel_depth = 3