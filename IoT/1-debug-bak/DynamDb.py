from __future__ import print_function

import json
import boto3
from decimal import Decimal

print('Loading function')
dynamodb = boto3.resource('dynamodb') 
table = dynamodb.Table('traffic_stat')
jstat = [{'id':1, 'speed':2, 'count':3, 'Enabled':False,'long':1.9},
{'id':4, 'speed':5, 'count':6, 'Enabled':False,'long':2.9}]

s_stat = json.dumps(jstat)
table.put_item(Item={'city': 'Aarhus', 'traffic_stat': s_stat})

print("Table status:", table.table_status)


def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    response = table.get_item(Key={'city': 'Aarhus'})
    item = response['Item']
    print("GetItem succeeded:")
    print (len(item),type(item))
    print (item)
    stat = item['traffic_stat']
    print (type(stat))
    stat = str(stat)
    print (stat)
    j_stat = json.loads(str(stat))
    print (j_stat[0]['long'])
    #print(json.dumps(item, indent=4))
    return event['key1']  # Echo back the first key value
    #raise Exception('Something went wrong')
