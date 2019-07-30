"""
Usage:

About: module to help with creating database tables for other module in package

Author: Satish Jasthi
"""
import numpy as np
from sqlalchemy import create_engine
from pymongo import MongoClient
from sqlalchemy_utils.functions import database_exists, create_database

from drshti_yantrikarana.config import db_password, db_username, db_name

mysql_credentials = f'mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}'

# MySQL db..............................................................................................................
# check if db exists else create new one
if database_exists(mysql_credentials):
    mysql_engine = create_engine(mysql_credentials)
else:
    mysql_engine = create_database(mysql_credentials)

# num classes
num_classes = np.squeeze(mysql_engine.execute('SELECT count(distinct(`Class`))'
                                              ' FROM drshit_yantrikarana.Class_wise_image_count').fetchall())

# Mongo db..............................................................................................................
mongo_client = MongoClient()
mongo_db = mongo_client['drshti_yantrikarana']
mean_std_coltn = mongo_db['mean_std']