from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from threading import Thread 
import paho.mqtt.client
import paho.mqtt.publish
import time
import requests
import json
import pandas as pd


def paho_on_connect(client, userdata, flags, rc):
    print "paho qtt connected with result code: "+str(rc)
    print "-------------------------\n"
    client.subscribe('Android/Aarhus/admin')
    #client.subscribe('AWS/Aarhus/debug')

def paho_on_message(client, userdata, msg):
    global AWS_Client
    print "paho client received a new message: "
    print 'msg.topic: '+str(msg.topic)
    print 'msg.payload: '+str(msg.payload)
    print("-------------------------\n")
    #if not msg.topic[:4] == 'NZL/':
    #    msg.topic = msg.topic[4:]
    #payload=json.loads(msg.payload)
    AWS_Client.publish(msg.topic, msg.payload, 0)


def AWS_Callback(client, userdata, msg):
    print("AWS client received a new message: ")
    print("msg topic: " + msg.topic)
    print("msg payload: " + msg.payload)
    print("-------------------------\n")

    paho.mqtt.publish.single(msg.topic, 
                        msg.payload, 
                        hostname='iot.eclipse.org')

def paho_run(paho_client):
    paho_client.on_connect = paho_on_connect
    paho_client.on_message = paho_on_message
    paho_client.connect('iot.eclipse.org',1883,60)
    paho_client.loop_forever()
    #while True:
    #    time.sleep(1.0)

AWS_Client = AWSIoTMQTTClient("HP-Ireland-app2")
endpoint = "a2mxpvymzj1qjd.iot.eu-west-1.amazonaws.com"
AWS_Client.configureEndpoint(endpoint, 8883)
root_ca = './keys/root-CA.crt'
prvt_key = './keys/HP-Ireland.private.key'
cert_pem = './keys/HP-Ireland.cert.pem'
AWS_Client.configureCredentials(root_ca, 
                                prvt_key, 
                                cert_pem)
AWS_Client.connect()
#myMQTTClient.subscribe("AWS/Aarhus/traffic/alert", 0, customCallback)
#myMQTTClient.subscribe("AWS/Aarhus/debug", 0, customCallback)
#myMQTTClient.subscribe("AWS/Aarhus/info", 0, customCallback)
AWS_Client.subscribe("AWS/Aarhus/#", 0, AWS_Callback)
#myMQTTClient.subscribe("PI/DHT11", 0, customCallback)
#AWS_Client.subscribe("$aws/things/PI_Ireland/shadow/#", 0, AWS_Callback)

paho_client = paho.mqtt.client.Client()
paho_thread = Thread(target=paho_run,
                    args=(paho_client,))

paho_thread.start()

print 'waiting for qtt message...'
idx = 0
while True:
    set_stat = {
        "state": {
        "reported": {"humi": 99, "color":None, "test":None},
        "desired" : None
        }, 
        #"version" : 69    
    }
    str_set_stat=json.dumps(set_stat)
    #AWS_Client.publish('traffic_stat/admin', str_set_stat, 0)
    #AWS_Client.publish('$aws/things/PI_Ireland/shadow/update', str_set_stat, 0)
    time.sleep(15)
    #AWS_Client.publish('$aws/things/PI_Ireland/shadow/get', '', 0)
    idx += 1
    time.sleep(15)
