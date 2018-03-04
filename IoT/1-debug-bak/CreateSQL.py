from __future__ import print_function

import os
from datetime import datetime
from urllib2 import urlopen
import boto3
import json
#import requests
import sys
import logging
import rds_config
import pymysql
#rds settings
rds_host = "rds-instance-endpoint"
name = rds_config.db_username
password = rds_config.db_password
db_name = rds_config.db_name
logger = logging.getLogger()
logger.setLevel(logging.INFO)
try:
	conn = pymysql.connect(rds_host, user=name, 
							passwd=password, db=db_name,
							connect_timeout=5)
except:
	logger.error("ERROR: Unexpected error: Could not connect to MySql instance.")
	sys.exit()
	logger.info("SUCCESS: Connection to RDS mysql instance succeeded")


client = boto3.client('iot-data', region_name='eu-west-1')
qttMsg = json.dumps([{"id":"1", "time_stamp": "2017-11-07T05:55:00",
                          "vehicle_count": 10, "vehicle_speed": 23},
                         {"id":"2", "time_stamp": "2017-11-07T05:55:00",
                          "vehicle_count": 20, "vehicle_speed": 30}])
        
dynamodb = boto3.resource('dynamodb') 


print("Table status:", table.table_status)

def lambda_handler(event, context):
    #print('Checking {} at {}...'.format(SITE, event['time']))
    client.publish(topic='AWS/Aarhus_admin', qos=0, payload=qttMsg)
    print('Check complete at {}'.format(str(datetime.now())))

