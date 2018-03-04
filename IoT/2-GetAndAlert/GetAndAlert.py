from __future__ import print_function
import boto3
import pandas as pd
import time
import requests
import json
import pandas as pd
import numpy as np

AAHUS_ALL_ID_FLAG = -1
AARHUS_MAX_SPEED = 120    # global maximum speed to be alerted
AARHUS_SUCCESS = 'scheduled task done'
AARHUS_EVENT_ERROR = 'undefined event'
AARHUS_DATA_ERROR = 'sensor data and statistics error'
AARHUS_INPUT_ERROR = 'input command error'
AARHUS_COMMAND_ERROR = 'Your command not supported yet'
AARHUS_SOURCE_APIGATEWAY = 'aws.api.gateway'
AARHUS_SOURCE_ANDROID_QTT = 'android.qtt'
AARHUS_SOURCE_SCHEDULED = 'aws.events'
AARHUS_COMMAND_GET = 'get'
AARHUS_COMMAND_PUT = 'put'
AARHUS_STATUS_NORMAL = 'normal'
AARHUS_STATUS_ALERTED = 'alerted'

dynamodb = boto3.resource('dynamodb')
stat_table = dynamodb.Table('traffic_stat')
traffic_table = dynamodb.Table('Aarhus_traffic')
client = boto3.client('iot-data', region_name='eu-west-1')


def Get_Data_From_Url():
    data_url = '''http://www.odaa.dk/api/action/datastore_search'''
    "/api/3/action/datastore_search?resource_id=b3eeb0ff-c8a8-4824-99d6-e0a3747c8b0d"
    data_resource_id = 'resource_id=b3eeb0ff-c8a8-4824-99d6-e0a3747c8b0d'
    offsets = [100, 200, 300, 400]
    url = data_url + '?' + data_resource_id
    res = requests.get(url)
    info = json.loads(str(res.text))
    stream_data = info['result']['records']
    for ofs in offsets:
        url = data_url+'?offset='+str(ofs)+'&'+data_resource_id
        res = requests.get(url)
        info = json.loads(str(res.text))
        stream_data = stream_data+info['result']['records']
    return stream_data


def GetAlertIDsUpdateStat(df_traffic, df_stat):
# df_stat: [REPORT_ID, speed_max, count_max, 
# speed_gate, count_gate, long1, long2, lat2, record_time]
# df_traffic: inherit from the meta_data, which is: 
# [REPORT_ID, avgSpeed, vehicleCount, TIMESTAMP, ...]
    AlertIDs = []

    for idx in range(len(df_traffic)):
        bAlert = False
        if float(df_stat['speed_max'][idx]) < float(df_traffic['avgSpeed'][idx]):
            df_stat.loc[idx, ['speed_max']] = df_traffic.iloc[idx]['avgSpeed']
            bAlert = True

        if df_stat.iloc[idx]['count_max'] < df_traffic.iloc[idx]['vehicleCount']:
            df_stat.loc[idx, ['count_max']] = df_traffic.iloc[idx]['vehicleCount']
            bAlert = True

        if df_traffic.iloc[idx]['avgSpeed'] > AARHUS_MAX_SPEED:
            bAlert = True

        if float(df_stat['speed_gate'][idx]) < float(df_traffic['avgSpeed'][idx]):
            bAlert = True

        if (df_stat.iloc[idx]['count_gate']) < (df_traffic.iloc[idx]['vehicleCount']):
            bAlert = True

        if bAlert:
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            Msg = {'REPORT_ID': df_traffic.loc[idx]['REPORT_ID'],
                   'speed': df_traffic.loc[idx]['avgSpeed'],
                   'count': df_traffic.loc[idx]['vehicleCount'],
                   'speed_max': df_stat.loc[idx]['speed_max'],
                   'count_max': df_stat.loc[idx]['count_max'],
                   'speed_gate': df_stat.loc[idx]['speed_gate'],
                   'count_gate': df_stat.loc[idx]['count_gate'],
                   'long1': df_stat.loc[idx]['long1'],
                   'lat1': df_stat.loc[idx]['lat1'],
                   'long2': df_stat.loc[idx]['long2'],
                   'lat2': df_stat.loc[idx]['lat2'],
                   'record_time': time_now
                   }
            AlertIDs.append(Msg)
            bAlert = False

    return AlertIDs


def GetIDInfo(id):
    response = stat_table.get_item(Key={'city': 'Aarhus'})
    item = response['Item']
    lst_stat = json.loads(item['traffic_stat'])
    lst_traffic = json.loads(item['latest_straffic'])

    str_alertIDs = item['latest_alert']
    id = int(id)
    if id == AAHUS_ALL_ID_FLAG:
        return str_alertIDs

    lst_alertIDs = json.loads(str_alertIDs)

    record_time = item['record_time']
    bTrafficFound = False  
    bStatFound = False
    status = AARHUS_STATUS_NORMAL
    time = ''
    speed = 0
    count = 0
    speed_max = 0
    count_max = 0
    speed_gate = 0
    count_gate = 0

    for idx in range(len(lst_traffic)):
        if id == lst_traffic[idx]['REPORT_ID']:
            bTrafficFound = True
            timestamp = lst_traffic[idx]['TIMESTAMP']
            speed = lst_traffic[idx]['avgSpeed']
            count = lst_traffic[idx]['vehicleCount']
            break

    for idx in range(len(lst_stat)):
        if id == lst_stat[idx]['REPORT_ID']:
            bStatFound = True
            speed_max = lst_stat[idx]['speed_max']
            count_max = lst_stat[idx]['count_max']
            speed_gate = lst_stat[idx]['speed_gate']
            count_gate = lst_stat[idx]['count_gate']
            break

    for idx in range(len(lst_alertIDs)):
        if id == lst_alertIDs[idx]['REPORT_ID']:
            status = AARHUS_STATUS_ALERTED
            break
    if bTrafficFound and bStatFound:  
    # to compatible with older verison of statistics table with id1164
        Msg = {'REPORT_ID': id,
               'TIMESTAMP': timestamp,
               'avgSpeed': speed,
               'vehicleCount': count,
               'speed_max': speed_max,
               'count_max': count_max,
               'speed_gate': speed_gate,
               'count_gate': count_gate,
               'status': status
              }
        Msg = json.dumps(Msg)
        return Msg
    else:
        return 'Not found this id'


def PutIDStatInfo(id, para):
    response = stat_table.get_item(Key={'city': 'Aarhus'})
    item = response['Item']
    lst_stat = json.loads(item['traffic_stat'])
    str_alertIDs = item['latest_alert']
    str_traffic = item['latest_straffic']

    id = int(id)
    if id == AAHUS_ALL_ID_FLAG:
        for idx in range(len(lst_stat)):
            lst_stat[idx]['speed_gate'] = para['speed_gate']
            lst_stat[idx]['count_gate'] = para['count_gate']
        df_stat = pd.DataFrame(lst_stat)
        str_stat = df_stat.to_json(orient='records')
        
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        stat_table.put_item(Item={'city': 'Aarhus', 
                                  'traffic_stat': str_stat,
                                  'latest_straffic': str_traffic,
                                  'latest_alert': str_alertIDs,
                                  'record_time': time_now})
        return 'Updated ' + str(len(lst_stat)) + ' records!'

    for idx in range(len(lst_stat)):
        if id == lst_stat[idx]['REPORT_ID']:
            lst_stat[idx]['speed_gate'] = para['speed_gate']
            lst_stat[idx]['count_gate'] = para['count_gate']

            df_stat = pd.DataFrame(lst_stat)
            str_stat = df_stat.to_json(orient='records')
            
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            stat_table.put_item(Item={'city': 'Aarhus', 
                                      'traffic_stat': str_stat,
                                      'latest_straffic': str_traffic,
                                      'latest_alert': str_alertIDs,
                                      'record_time': time_now})
            Msg = {'REPORT_ID': id,
                   'speed_gate': para['speed_gate'],
                   'count_gate': para['count_gate']}
            Msg = json.dumps(Msg)
            Msg = 'Updated ID: ' + Msg
            return Msg

    return AARHUS_INPUT_ERROR + ', ID not found'
    

def lambda_handler(event, context):
    str_event = 'Nzl recved event info: '+json.dumps(event)
    #print (str_event)
    client.publish(topic='AWS/Aarhus/debug', 
                   qos=0, payload=str_event)

    if AARHUS_SOURCE_APIGATEWAY == event['source'].lower():
        if AARHUS_COMMAND_GET == event['command'].lower():
            id = int(event['id'])
            Msg = GetIDInfo(id)
            return Msg

        elif AARHUS_COMMAND_PUT == event['command'].lower():            
            if (not event['speed_gate'].isdigit() or \
                not event['count_gate'].isdigit()):
               return AARHUS_INPUT_ERROR  

            id = int(event['id'])
            para = {'speed_gate': int(event['speed_gate']),
                    'count_gate': int(event['count_gate'])}
            Msg = PutIDStatInfo(id, para)
            return Msg

        else:
            return AARHUS_COMMAND_ERROR

    elif AARHUS_SOURCE_SCHEDULED == event['source'].lower():
        response = stat_table.get_item(Key={'city': 'Aarhus'})
        item = response['Item']
        str_stat = item['traffic_stat']
        lst_stat = json.loads(str_stat)
        traffic_data = Get_Data_From_Url()

        df_traffic = pd.DataFrame(traffic_data)
        df_stat = pd.DataFrame(lst_stat)

        df_traffic = df_traffic.sort_values(by='REPORT_ID')
        df_stat = df_stat.sort_values(by='REPORT_ID')
        df_traffic = df_traffic.reset_index(drop=True)
        df_stat = df_stat.reset_index(drop=True)

        #debug for change the older verion statistics table
        #idx = 0 
        #for idx in range(len(df_stat)):
        #    if 1164 == df_stat['REPORT_ID'][idx]:
        #        break;
        #client.publish(topic='AWS/Aarhus/traffic/debug', 
        #               qos=0, payload=str(idx))
        #df_stat = df_stat.drop(df_traffic.index[idx])
        #df_stat = df_stat.sort_values(by='REPORT_ID')
        #df_stat = df_stat.reset_index(drop=True)
        #debug for change the older verion statistics table

        #In case the meta data changes
        ser = np.abs(df_traffic['REPORT_ID'] - df_stat['REPORT_ID'])
        if ser.sum() != 0:
            df_debug['id_tr'] = df_traffic['REPORT_ID']
            df_debug['id_stat'] = df_stat['REPORT_ID']
            str_df_debug = df_debug.to_json(orient='records')
            client.publish(topic='AWS/Aarhus/debug', qos=0, 
                           payload='Data Error: '+str(ser.sum()))
            client.publish(topic='AWS/Aarhus/debug', qos=0, 
                           payload=str_df_debug)
            return AARHUS_DATA_ERROR
        #For in case the meta data changes

        AlertIDs = GetAlertIDsUpdateStat(df_traffic, df_stat)
        df_AlertIDs = pd.DataFrame(AlertIDs)
        str_AlertIDs = df_AlertIDs.to_json(orient='records')
        AlertMsg = {'Alert_ID_Num': len(AlertIDs),
                    'Content': str_AlertIDs}
        AlertMsg = json.dumps(AlertMsg)
        
        #print (len(AlertIDs))
        str_stat = df_stat.to_json(orient='records')
        str_traffic = df_traffic.to_json(orient='records')
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        stat_table.put_item(Item={'city': 'Aarhus', 
                                  'traffic_stat': str_stat,
                                  'latest_straffic': str_traffic,
                                  'latest_alert': str_AlertIDs,
                                  'record_time': time_now})
        #traffic_table.put_item(Item={'time': time_now, 
        #                             'traffic': str_traffic,
        #                             'alert_ids': str_AlertIDs})
        # to save DB space, not save the raw dat any more
        client.publish(topic='AWS/Aarhus/traffic/alert', 
                       qos=0, payload=AlertMsg)
        
        return AARHUS_SUCCESS

    elif AARHUS_SOURCE_ANDROID_QTT == event['source'].lower():
        if (not 'command' in event.keys() or \
            not 'id' in event.keys()):
            client.publish(topic='AWS/Aarhus/info', 
                           qos=0, payload=AARHUS_INPUT_ERROR)
            return AARHUS_INPUT_ERROR

        if AARHUS_COMMAND_GET == event['command'].lower():
            id = int(event['id'])
            Msg = GetIDInfo(id)
            client.publish(topic='AWS/Aarhus/info', 
                           qos=0, payload=Msg)
            return AARHUS_SUCCESS
            
        else:
            client.publish(topic='AWS/Aarhus/debug', 
                           qos=0, payload=AARHUS_COMMAND_ERROR)
            return AARHUS_COMMAND_ERROR

    else:
        client.publish(topic='AWS/Aarhus/debug', 
                       qos=0, payload=AARHUS_EVENT_ERROR)
        return AARHUS_EVENT_ERROR
