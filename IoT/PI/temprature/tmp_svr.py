import paho.mqtt.client as mqtt
from threading import Thread
from flask import Flask, render_template

temperature = 0
max_temp = 0
min_temp = 99
humidity = 0
max_humi = 0
min_humi = 99

def on_connect(client, userdata, flags, rc):
	print('Connected with result code: '+str(rc))
	client.subscribe('nzl_data/temperature')
	client.subscribe('nzl_data/humidity')

def on_message(client, userdata, msg):
	global temperature, max_temp, min_temp
	global humidity, max_humi, min_humi

	print str(msg.topic)+': '+str(msg.payload)
	if(msg.topic=='nzl_data/temperature'):
		try:
			temperature = float(msg.payload)
		except:
			pass
		if temperature > max_temp:
			max_temp = temperature
		if temperature < min_temp:
			min_temp = temperature

	if(msg.topic=='nzl_data/humidity'):
		humidity = float(msg.payload)
		if humidity > max_humi:
			max_humi = humidity
		if humidity < min_humi:
			min_humi = humidity

	print max_temp, min_temp, max_humi, min_humi
#def on_message(mqttc, obj, msg):
#	print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))

def qtt_run():
	client = mqtt.Client()
	client.on_connect = on_connect
	client.on_message = on_message
	client.connect('iot.eclipse.org',1883)
	client.loop_forever()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	global temperature, max_temp, min_temp
	global humidity, max_humi, min_humi

	return render_template('Index.html', 
							temp=temperature, 
							tempMax=max_temp, 
							tempMin=min_temp, 
							humi=humidity, 
							humiMax=max_humi, 
							humiMin=min_humi)

if __name__ == '__main__':
	qtt_th = Thread(target=qtt_run)
	qtt_th.start()
	app.run(host='0.0.0.0', debug=True, port=int(8000))
