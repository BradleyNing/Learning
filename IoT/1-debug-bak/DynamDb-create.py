from __future__ import print_function

import json
import boto3

print('Loading function')
dynamodb = boto3.resource('dynamodb') 
table = dynamodb.create_table(
    TableName='Aarhus_traffic_stat',
    KeySchema=[
        {
        'AttributeName': 'time',
        'KeyType': 'HASH' #Partition key
        }],
    AttributeDefinitions=[
        {
        'AttributeName': 'time',
        'AttributeType': 'S'
        }],
    ProvisionedThroughput={
        'ReadCapacityUnits': 1,
        'WriteCapacityUnits': 1
        }
    )

#table.put_item(Item={'year': 2017, 'title': 'Nzl', 'info': 'Nzl Test',})
table.put_item(Item={'year': 2017, 'title': 'Nzl', 
    'id': 100,
    'speed': 100,
    'count': 90,
    'Enabled': True,
    'long': 19.9})


print("Table status:", table.table_status)
