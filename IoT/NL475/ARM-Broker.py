from flask import Flask, request, render_template
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from pytz import timezone
import datetime
import json
import logging
logging.basicConfig(level=logging.INFO)

def customCallback(client, userdata, message):
    global bWait, MsgNum
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

app = Flask(__name__)
@app.route('/', methods=['GET'])
def test():
    if 'idx' in request.args:
        # idx = 'idx: '+request.args['idx']
        # temp = 'temp: '+request.args['temp']
        # humi = 'humi: '+request.args['humi']
        # pressure = 'pressure: '+request.args['pressure']
        # msg = idx+', '+temp+', '+humi+', '+pressure
        # now = datetime.datetime.now(timezone('Europe/Dublin'))
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%dT%H:%M:%S')
        data = {
                "idx": int(request.args['idx']),
                "datetime": now_str,
                "temperature(C)": float(request.args['temp']),
                "humidity(%)": float(request.args['humi']),
                "pressure(mBar)": float(request.args['pressure'])
                }
        data_str = json.dumps(data)
        #logging.info(msg)
        qttClient.publish("arm/data", data_str, 0)
        return 'sucess'
    else:
        return '?'

@app.route('/rpt', methods=['POST'])
def rpt():
    req = request.values
    logging.info(str(req[0]))
    return 'Hello from POST!'

import traceback
PORT=9000
if __name__ == '__main__':
    try:
        qttClient = AWSIoTMQTTClient("STM32L4-Broker")
        endpoint = "a2mxpvymzj1qjd.iot.eu-west-1.amazonaws.com"
        qttClient.configureEndpoint(endpoint, 8883)
        root_ca = 'keys/root-CA.crt'
        p_key = 'keys/PI-Ireland-private.key'
        cert = 'keys/PI-Ireland-cert.pem'
        qttClient.configureCredentials(root_ca,
                                       p_key,
                                       cert)
        qttClient.configureConnectDisconnectTimeout(1000)  # 1000 sec
        qttClient.connect()
        #qttClient.subscribe("AWS/Aarhus/traffic/alert", 0, customCallback)
        #app.run(debug=True, port=int(PORT))
        #app.run(debug=True, host='0.0.0.0', port=int(PORT))
        app.run(host='0.0.0.0', port=int(PORT))
    except:
        traceback.print_exc()
