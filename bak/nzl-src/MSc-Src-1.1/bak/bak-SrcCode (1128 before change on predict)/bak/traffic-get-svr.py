from flask import Flask, jsonify, abort, make_response, request
from flask_httpauth import HTTPBasicAuth
from threading import Thread
import pandas as pd
import requests
import json
import sqlite3
import time
import traceback
import logging
import DAAS_DEF
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
auth = HTTPBasicAuth()
bCollecting = False
end_time = None
data_url = '''http://www.odaa.dk/api/action/datastore_search'''
data_resource_id = 'resource_id=b3eeb0ff-c8a8-4824-99d6-e0a3747c8b0d'
db_name = "traffic_data_from201709.sqlite3"
#db_conn = sqlite3.connect(db_name)

@auth.get_password
def get_password(username):
    if username == 'Bradley-ZN00046':
        return 'Bradley-ZN00046'
    return None

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'response': 'Bad request'}), 400)

def Get_Data_From_Url():
    global data_url, data_resource_id

    offsets = [100,200,300,400]
    url = data_url+'?'+data_resource_id
    res=requests.get(url)
    info = json.loads(str(res.text))
    stream_data = info['result']['records']
    for ofs in offsets:
        url = data_url+'?offset='+str(ofs)+'&'+data_resource_id
        res=requests.get(url)
        info = json.loads(str(res.text))
        stream_data = stream_data+info['result']['records']
    return stream_data

def UpdateDb(db_conn_th, sd):
    if len(sd) == 0:
        return
    for idx in range(0, len(sd)):
        rpt_id = sd[idx]['REPORT_ID']
        tableName = 'trafficData'+str(rpt_id)
        sqlTxt ='INSERT INTO '+tableName+'''(status, 
                        avgMeasuredTime, 
                        avgSpeed, 
                        medianMeasuredTime, 
                        TIMESTAMP, 
                        vehicleCount, 
                        _id, 
                        REPORT_ID) VALUES (?,?,?,?,?,?,?,?)'''
        db_conn_th.execute(sqlTxt,(sd[idx]['status'], 
                        sd[idx]['avgMeasuredTime'], 
                        sd[idx]['avgSpeed'],
                        #sd[idx]['extID'],
                        sd[idx]['medianMeasuredTime'],
                        sd[idx]['TIMESTAMP'], 
                        sd[idx]['vehicleCount'], 
                        sd[idx]['_id'], 
                        sd[idx]['REPORT_ID']))

    db_conn_th.commit()

def Get_Data_Thread():
    global bCollecting
    #global end_time
    bInTime = True
    logging.info('In thread')
    db_conn_th = sqlite3.connect(db_name)
    while bCollecting:
        sd = Get_Data_From_Url()
        UpdateDb(db_conn_th, sd)
        time.sleep(5.0*60) #5 minutes
    db_conn_th.close()

def Data_Feed(rpt_id,latest_n):
    db_conn_read = sqlite3.connect(db_name)

    tableName = 'trafficData'+str(rpt_id)
    sqlTxt = 'SELECT * FROM ' + tableName
    df = pd.read_sql(sqlTxt, db_conn_read)
    records = list()

    for i in range(1,latest_n+1):
        item = {'status':str(df['status'].iloc[-i]),
                'avgMeasuredTime':df['avgMeasuredTime'].iloc[-i],
                'avgSpeed':df['avgSpeed'].iloc[-i],
                'medianMeasuredTime':df['medianMeasuredTime'].iloc[-i],
                'TIMESTAMP':str(df['TIMESTAMP'].iloc[-i]),
                'vehicleCount':df['vehicleCount'].iloc[-i],
                'REPORT_ID':df['REPORT_ID'].iloc[-i]
                }
        records.append(item)
        result = {'records':records}
    return result

@app.route('/', methods=['POST'])
@auth.login_required
def start_stop():
    global bCollecting
    #global end_time

    req = request.json    
    if req == None:
        abort(400)

    logging.info(str(req))
    if req['type'] == DAAS_DEF.start :
        if bCollecting == False:
            #end_time = req['para']['end_time']
            #end_time = datetime.strptime(end_time, '%Y-%m-%d-%H-%M-%S')
            get_data_thread = Thread(target=Get_Data_Thread)
            get_data_thread.start()
            bCollecting = True
            result = {'start':DAAS_DEF.done}
        else:
            result = {'start':DAAS_DEF.duplicate_start}
        return jsonify({'response': result}), 201

    elif req['type']==DAAS_DEF.stop :
        if bCollecting == True:
            bCollecting = False
            result = {'stop':DAAS_DEF.done}
        else:
            result = {'stop':DAAS_DEF.duplicate_stop}
        return jsonify({'response': result}), 201

    elif req['type']==DAAS_DEF.get_data:
        report_id=req['para']['report_id']
        latest_n = req['para']['latest_n']  
        result = Data_Feed(report_id,latest_n)
        #logging.info(str(result))
        result = json.dumps(str(result))
        return jsonify({'response': result}), 201   

    else:
        abort(400)


TRAFFIC_GET_PORT = 9999
if __name__ == '__main__':
    try:
        app.run(debug=True, port=TRAFFIC_GET_PORT)
        #app.run(debug=True, host='0.0.0.0', port=TRAFFIC_GET_PORT)
    except:
        traceback.print_exc()
