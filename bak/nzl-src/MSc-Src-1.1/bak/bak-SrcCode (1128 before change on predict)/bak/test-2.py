import sqlite3
import pandas as pd
import json

def dict2json(d):
    return Student(d['name'], d['age'], d['score'])

db_name = 'traffic_data_from201709.sqlite3'
db_conn_read = sqlite3.connect(db_name)

tableName = 'trafficData'+str(158415)
sqlTxt = 'SELECT * FROM ' + tableName
df = pd.read_sql(sqlTxt, db_conn_read)
records = []
latest_n=1
#print df.info()

for i in range(1,latest_n+1):
    item = {df['avgSpeed'].iloc[-i],
            df['vehicleCount'].iloc[-i],
            df['REPORT_ID'].iloc[-i],
            str(df['TIMESTAMP'].iloc[-i])}
    #item = {"status": str(df['status'].iloc[-i]),
    #        "avgMeasuredTime": df['avgMeasuredTime'].iloc[-i],
    #        "avgSpeed":df['avgSpeed'].iloc[-i],
    #        "medianMeasuredTime": df['medianMeasuredTime'].iloc[-i],
    #        "TIMESTAMP": str(df['TIMESTAMP'].iloc[-i]),
    #        "vehicleCount": df['vehicleCount'].iloc[-i],
    #        "REPORT_ID":df['REPORT_ID'].iloc[-i]
    #        }
    records.append(item)
result = {"records":records}

print type(result)
print result