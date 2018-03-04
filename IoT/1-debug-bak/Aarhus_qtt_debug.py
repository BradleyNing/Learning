from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import requests
import json
import pandas as pd
import numpy as np
import datetime

def customCallback(client, userdata, msg):
    print("Received a new message: ")
    print("Msg topic: " + msg.topic)
    print("Msg payload: " + msg.payload)
    print("-----------------------------\n")

myMQTTClient = AWSIoTMQTTClient("HP-Ireland")
endpoint = "a2mxpvymzj1qjd.iot.eu-west-1.amazonaws.com"
myMQTTClient.configureEndpoint(endpoint, 8883)
root_ca = '../keys/root-CA.crt'
prvt_key = '../keys/HP-Ireland.private.key'
cert_pem = '../keys/HP-Ireland.cert.pem'
myMQTTClient.configureCredentials(root_ca, 
                                  prvt_key, 
                                  cert_pem)
myMQTTClient.connect()
#myMQTTClient.subscribe("AWS/Aarhus/traffic/alert", 0, customCallback)
myMQTTClient.subscribe("AWS/Aarhus/debug", 0, customCallback)
myMQTTClient.subscribe("AWS/Aarhus/traffic/debug", 0, customCallback)
myMQTTClient.subscribe("AWS/Aarhus/info", 0, customCallback)
#myMQTTClient.subscribe("AWS/Aarhus/#", 0, customCallback)
#myMQTTClient.subscribe("PI/DHT11", 0, customCallback)
#myMQTTClient.subscribe("$aws/things/PI_Ireland/shadow/#", 0, customCallback)
#myMQTTClient.subscribe("AWS/dht11/#", 0, customCallback)


print 'waiting for qtt message...'
idx = 0
while True:
    #set_stat = {
        #"state": {
        #"reported": {"blinking":'red'},
        #"desired" : {"blinking":'red'}
        #}, 
        #"version" : 69    
    #}
    temperature = np.random.randint(0, 30)
    humidity = np.random.randint(0, 99)
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%dT%H:%M:%S')
    
    data = {"datetime": now_str,
            "temperature": temperature,
            "humidity": humidity}
    data_str = json.dumps(data)
    #print data
    #myMQTTClient.publish('dht11/data', data_str, 0)
    #myMQTTClient.publish('$aws/things/PI_Ireland/shadow/get', '', 0)
    #myMQTTClient.publish('traffic_stat/admin', str_set_stat, 0)
    time.sleep(10)
    #myMQTTClient.publish('$aws/things/PI_Ireland/shadow/update', str_set_stat, 0)
    idx += 1
    #time.sleep(5)
