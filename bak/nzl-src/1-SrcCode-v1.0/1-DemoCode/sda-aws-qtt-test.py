from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
MsgNum = 0
bWait = True

def customCallback(client, userdata, message):
    global MsgNum, bWait
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")
    MsgNum +=1
    if MsgNum==5:
        bWait = False

myMQTTClient = AWSIoTMQTTClient("HP")
host = "a2mxpvymzj1qjd.iot.eu-west-2.amazonaws.com"
myMQTTClient.configureEndpoint(host, 8883)

myMQTTClient.configureCredentials("static/root-CA.crt", 
                                  "static/HP-PC.private.key", 
                                  "static/HP-PC.cert.pem")

myMQTTClient.connect()
myMQTTClient.publish("daas", "response from daas:...", 1)
myMQTTClient.subscribe("daas/SDA", 1, customCallback)
myMQTTClient.subscribe("daas/SDA_alert", 1, customCallback)
myMQTTClient.subscribe("daas/SDA", 0, customCallback)
myMQTTClient.subscribe("daas/SDA_alert", 0, customCallback)

print 'publishing for 5 qtt msgs'
pubNum = 5
for i in range(pubNum):
    #myMQTTClient.publish("daas/SDA_alert", "Publish message from HP: "+str(i), 1)
    time.sleep(1)

print 'Wait for daas/SDA msg ...'
bWait = True
while bWait:
    time.sleep(1)