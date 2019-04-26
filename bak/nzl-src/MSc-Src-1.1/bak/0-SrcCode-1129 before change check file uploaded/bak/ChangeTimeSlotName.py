import sqlite3
import os
import pandas as pd
import time

db_folder = 'C:\\1-Programming\\1-MSC-Test\\TestDB'
db_name = os.path.join(db_folder, "WholeDB0802.sqlite3")
con = sqlite3.connect(db_name)
sqlTxt = 'SELECT REPORT_ID FROM SensorMetaDataTable'
a_rpt_id = pd.read_sql(sqlTxt, con)
a_rpt_id = a_rpt_id.values
count = 0
s_time = time.time()
for id in a_rpt_id:
	rid = id[0]
	if rid == 1164:
		continue

	tableName = 'vehicleCount'+str(rid)
	sqlTxt = 'SELECT * FROM '+ tableName
	print tableName
	df = pd.read_sql(sqlTxt, con)
	print df['slot_type']
	
	s_ts = ['Peak1', 'Peak2', 'Peak3', 'Offpeak1', 'Offpeak2', 'WEH1', 'WEH2']
	#df['slot_type'] = s_ts
	break
	#df['slot_type'][0]='Peak1'
	#if df['slot_type']=='p2':

	df.to_sql(tableName, con, if_exists='replace', index=False)
	#sqlTxt = 'UPDATE '+tableName+''' SET slot_type='Peak1' WHERE slot_type='p1';'''
	count +=1
	print count

con.commit()
con.close()
e_time = time.time()
print(e_time-s_time)


