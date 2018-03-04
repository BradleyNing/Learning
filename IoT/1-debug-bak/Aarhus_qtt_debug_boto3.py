from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import requests
import json
import pandas as pd

def customCallback(client, userdata, msg):
    print("Received a new message: ")
    print("Msg topic: " + msg.topic)
    print("Msg payload: " + msg.payload)
    print("-----------------------------\n")

myMQTTClient = AWSIoTMQTTClient("HP-Ireland-app1")
endpoint = "a2mxpvymzj1qjd.iot.eu-west-1.amazonaws.com"
myMQTTClient.configureEndpoint(endpoint, 8883)
root_ca = './keys/root-CA.crt'
prvt_key = './keys/HP-Ireland.private.key'
cert_pem = './keys/HP-Ireland.cert.pem'
myMQTTClient.configureCredentials(root_ca, 
                                  prvt_key, 
                                  cert_pem)
myMQTTClient.connect()
#myMQTTClient.subscribe("AWS/Aarhus/traffic/alert", 0, customCallback)
#myMQTTClient.subscribe("AWS/Aarhus/debug", 0, customCallback)
#myMQTTClient.subscribe("AWS/Aarhus/info", 0, customCallback)
myMQTTClient.subscribe("AWS/Aarhus/#", 0, customCallback)
#myMQTTClient.subscribe("PI/DHT11", 0, customCallback)
#myMQTTClient.subscribe("$aws/things/PI_Ireland/shadow/#", 0, customCallback)

print 'waiting for qtt message...'
idx = 0
while True:
    set_stat = {
        "state": {
        #"reported": {"blinking":'red'},
        "desired" : {"blinking":'green'}
        }, 
        #"version" : 69    
    }
    str_set_stat=json.dumps(set_stat)
    #myMQTTClient.publish('$aws/things/PI_Ireland/shadow/get', '', 0)
    #myMQTTClient.publish('traffic_stat/admin', str_set_stat, 0)
    time.sleep(5)
    #myMQTTClient.publish('$aws/things/PI_Ireland/shadow/update', str_set_stat, 0)
    idx += 1
    time.sleep(5)
