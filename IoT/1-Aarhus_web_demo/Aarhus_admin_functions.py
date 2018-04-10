import pandas as pd
import time
import json
import logging
logging.basicConfig(level=logging.INFO)

def CreateJsAlerted(alertSensors):
    df_all=pd.read_csv("static/metaSensorData-Download.csv")
    stime = time.strftime('%H-%M-%S',time.localtime(time.time()))
    mapfile_name = stime+'map_markers.js'
    whole_name = './static/'+mapfile_name
    fp=open(whole_name, 'w')
    fp.write('var markers = [\n')

    len_all = len(df_all)
    df_sensors = pd.DataFrame(alertSensors)
    len_alerted = len(df_sensors)
    IDs = df_sensors['REPORT_ID'].values

    for i in range(len_alerted):
        long1 = df_sensors['long1'][i]
        lat1 = df_sensors['lat1'][i]
        long2 = df_sensors['long2'][i]
        lat2 = df_sensors['lat2'][i]
        rid = df_sensors['REPORT_ID'][i]
        fp.write('{\n')
        fp.write('\t"color": "red",\n')
        fp.write('\t"reportId": '+str(rid)+',\n')
        fp.write('\t"lng": '+str(long1)+',\n')
        fp.write('\t"lat": '+str(lat1)+',\n')
        fp.write('},\n')

        fp.write('{\n')
        fp.write('\t"color": "red",\n')
        fp.write('\t"reportId": '+str(rid)+',\n')
        fp.write('\t"lng": '+str(long2)+',\n')
        fp.write('\t"lat": '+str(lat2)+',\n')
        fp.write('},\n')
    
    fp.write('];') 
    fp.close()
    return mapfile_name

def CreateJsAll(alertSensors):
    df_all=pd.read_csv("static/metaSensorData-Download.csv")
    stime = time.strftime('%H-%M-%S',time.localtime(time.time()))
    mapfile_name = stime+'map_markers.js'
    whole_name = './static/'+mapfile_name
    fp=open(whole_name, 'w')
    fp.write('var markers = [\n')

    len_all = len(df_all)
    df_sensors = pd.DataFrame(alertSensors)
    if len(alertSensors) == 0:
        IDs = []
    else:
        IDs = df_sensors['REPORT_ID'].values

    for i in range(len_all):
        long1 = df_all['POINT_1_LNG'][i]
        lat1 = df_all['POINT_1_LAT'][i]
        long2 = df_all['POINT_2_LNG'][i]
        lat2 = df_all['POINT_2_LAT'][i]

        fp.write('{\n')
        rid = int(df_all['REPORT_ID'][i])    
        if rid in IDs:
            fp.write('\t"color": "red",\n')
        else :
            fp.write('\t"color": "blue",\n')
        fp.write('\t"reportId": '+str(rid)+',\n')
        fp.write('\t"lng": '+str(long1)+',\n')
        fp.write('\t"lat": '+str(lat1)+',\n')
        fp.write('},\n')

        fp.write('{\n')
        if rid in IDs:
            fp.write('\t"color": "red",\n')
        else :
            fp.write('\t"color": "blue",\n')
        fp.write('\t"reportId": '+str(rid)+',\n')
        fp.write('\t"lng": '+str(long2)+',\n')
        fp.write('\t"lat": '+str(lat2)+',\n')
        #fp.write('\t"speed": '+str(speed)+',\n')
        #fp.write('\t"count": '+str(count)+',\n')
        fp.write('},\n')

    fp.write('];') 
    fp.close()
    return mapfile_name
#end of CreateJs