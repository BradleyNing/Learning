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
records = list()
latest_n=1
#print df.info()

for i in range(1,latest_n+1):
    item = {df['avgSpeed'].iloc[-i],df['vehicleCount'].iloc[-i],df['REPORT_ID'].iloc[-i]}
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

#result = json.dumps(str(result))
print type(result)
#print result
#print result['records'][0]['vehicleCount']

result = str(result)
#rint result
print type(result)
#result = json.loads(result)

print '\n'
#test = '''{"records": [{"status": "OK","speed": 2,"count": 3}]}'''
test = '''{"records": [{"status": "OK", "avgMeasuredTime": 354, "TIMESTAMP": "2017-09-14T14:20:00", 
"medianMeasuredTime": 354, "avgSpeed": 29, "vehicleCount": 15, "REPORT_ID": 158415}]}'''

test1 = '''{'records': [{'status': 'OK', 'avgMeasuredTime': 354, 'TIMESTAMP': '2017-09-14T14:20:00', 
'medianMeasuredTime': 354, 'avgSpeed': 29, 'vehicleCount': 15, 'REPORT_ID': 158415}]}'''
#print test
#print type(test)
test=json.loads(test)
#test1=json.dumps(test1)
print test1
test1.replace("'", '''"''')
print '\n'

print test1
#test1=json.loads(test1)


#print type(test1)
#print test1['records'][0]['vehicleCount']