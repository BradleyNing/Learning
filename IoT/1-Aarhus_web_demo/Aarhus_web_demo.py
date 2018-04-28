from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from Aarhus_admin_forms import Form_admin_get, Form_admin_put, Form_admin_show, Form_led_config
from Aarhus_admin_functions import CreateJsAlerted, CreateJsAll
import requests
import json
import logging
import boto3
logging.basicConfig(level=logging.INFO)

app=Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
Bootstrap(app)

LED_GATE = 10
qttClient = ''

def customCallback(client, userdata, msg):
    global qttClient, LED_GATE
    logging.info("Received a new message: ")
    logging.info("Msg topic: " + msg.topic)
    js_payload = json.loads(msg.payload)
    logging.info('Alert_ID_Num: ' + str(js_payload['Alert_ID_Num']))
    if int(js_payload['Alert_ID_Num']) > LED_GATE:
        set_state = {"state": {"desired": {"blinking":'red'}}}
        qttClient.publish('$aws/things/PI_Ireland/shadow/update', 
                       json.dumps(set_state), 0)    

    else:
        set_state = {"state": {"desired": {"blinking":'green'}}}
        qttClient.publish('$aws/things/PI_Ireland/shadow/update', 
                       json.dumps(set_state), 0)         

def getIDToken():
    region="eu-west-1"
    user_pool_id="eu-west-1_ll4kiNDSx"
    # User pool client id for app with Admin NO SRP Auth enabled
    client_id="2jerk948g12o2o2hqdgmfa2960"
     
    # API Gateway endpoint
    endpoint="https://14sg61sezg.execute-api.eu-west-1.amazonaws.com/test"
     
    # User login details
    username="Bradley"
    password="xsw2!QAZ"
    auth_data = {'USERNAME':username, 'PASSWORD':password }
     
    # Get JWT token for the user
    provider_client=boto3.client('cognito-idp', region_name=region)
    resp = provider_client.admin_initiate_auth(UserPoolId=user_pool_id, 
                                AuthFlow='ADMIN_NO_SRP_AUTH', 
                                AuthParameters=auth_data, 
                                ClientId=client_id)
    idToken = resp['AuthenticationResult']['IdToken']
    return idToken

@app.route('/', methods=['GET','POST'])
def admin():
    global qttClient
    formGet = Form_admin_get()
    formPut = Form_admin_put()
    formShow = Form_admin_show()
    formLed = Form_led_config()
    if formGet.submitGet.data:
        if formGet.validate_on_submit():
            res_info = ''
            url = 'https://14sg61sezg.execute-api.eu-west-1.amazonaws.com/test'
            path = '/aws.api.gateway'
            id = int(formGet.getID.data)
            queryStr = '?command=get&id='+str(id)
            qUrl = url+path+queryStr
            header = {'Content-Type': 'application/json'}
            #auth=HTTPBasicAuth('Bradley-ZN00046', 'Bradley-ZN00046')
            res = requests.get(qUrl, headers=header)
            res_info += 'status_code: '+str(res.status_code)+'\n'
            res_json = res.json()
            if id == -1 and res.status_code == 200:            
                lst_res = json.loads(res_json)
                a_len = len(lst_res)
                res_info += 'Get '+str(a_len)+' alerted points, '
            res_info += 'Content: '+ str(res_json) +'\n\n'

            formGet.getInfoText.data = res_info
            return render_template('Traffic_admin.html', 
                                    formGet=formGet,
                                    formPut=formPut,
                                    formShow=formShow,
                                    formLed=formLed)
    
    if formPut.submitPut.data:
        if formPut.validate_on_submit():
            res_info = ''
            url = 'https://14sg61sezg.execute-api.eu-west-1.amazonaws.com/test'
            id = int(formPut.putID.data)
            path = '/aws.api.gateway'
            queryStr = '?command=put&id='+str(id)
            qUrl = url+path+queryStr

            #idToken=getIDToken()
            #headers = {'Content-Type': 'application/json',
            #            'Authorization': idToken}
            header = {'Content-Type': 'application/json'}
            req_body = {'speed_gate':formPut.inputSpeedGate.data,
                        'count_gate':formPut.inputCountGate.data}
            #auth=HTTPBasicAuth('Bradley-ZN00046', 'Bradley-ZN00046')
            res = requests.put(qUrl, headers=header, json=req_body)
            res_info += 'status_code: '+str(res.status_code)+'\n'
            res = res.json()
            res_info += 'response content: '+ str(res) +'\n\n'

            formPut.putInfoText.data = res_info
            return render_template('Traffic_admin.html', 
                                    formGet=formGet,
                                    formPut=formPut,
                                    formShow=formShow,
                                    formLed=formLed)

    if formShow.showAllSensors.data:
        mapfile = 'sensors_markers.js'
        return render_template('showOnMap.html', mapfile=mapfile)

    if formShow.showAlertedSensors.data:
        url = 'https://14sg61sezg.execute-api.eu-west-1.amazonaws.com/test'
        path = '/aws.api.gateway'
        queryStr = '?command=get&id=-1'
        qUrl = url+path+queryStr
        header = {'Content-Type': 'application/json'}
        res = requests.get(qUrl, headers=header)
        if (res.status_code != 200): 
            return render_template('Traffic_admin.html', 
                            formGet=formGet,
                            formPut=formPut,
                            formShow=formShow,
                            formLed=formLed)
        res = res.json()
        logging.info('Debug: '+str(res))
        alertSensors = json.loads(str(res))
        if len(alertSensors) == 0 :
            mapfile =''
        else:
            mapfile = CreateJsAlerted(alertSensors)
        return render_template('showOnMap.html', mapfile=mapfile)

    if formShow.showTogether.data:
        url = 'https://14sg61sezg.execute-api.eu-west-1.amazonaws.com/test'
        path = '/aws.api.gateway'
        queryStr = '?command=get&id=-1'
        qUrl = url+path+queryStr
        header = {'Content-Type': 'application/json'}
        res = requests.get(qUrl, headers=header)
        if (res.status_code != 200): 
            return render_template('Traffic_admin.html', 
                            formGet=formGet,
                            formPut=formPut,
                            formShow=formShow,
                            formLed=formLed)
        res = res.json()
        alertSensors = json.loads(str(res))
        mapfile = CreateJsAll(alertSensors)
        return render_template('showOnMap.html', mapfile=mapfile)

    if formLed.submitRedLed.data:
        set_state = {"state": {
                      "desired": {"blinking":'red'}}}
        qttClient.publish('$aws/things/PI_Ireland/shadow/update', 
                       json.dumps(set_state), 0)

    if formLed.submitGreenLed.data:
        set_state = {"state": {
                      "desired": {"blinking":'green'}}}
        qttClient.publish('$aws/things/PI_Ireland/shadow/update', 
                       json.dumps(set_state), 0)

    return render_template('Traffic_admin.html', 
                            formGet=formGet,
                            formPut=formPut,
                            formShow=formShow,
                            formLed=formLed)

import traceback
PORT=8890
if __name__ == '__main__':  
    try:
        qttClient = AWSIoTMQTTClient("HP-Ireland-app1")
        endpoint = "a2mxpvymzj1qjd.iot.eu-west-1.amazonaws.com"
        qttClient.configureEndpoint(endpoint, 8883)
        root_ca = './keys/root-CA.crt'
        prvt_key = './keys/HP-Ireland.private.key'
        cert_pem = './keys/HP-Ireland.cert.pem'
        qttClient.configureCredentials(root_ca, 
                                       prvt_key, 
                                       cert_pem)
        #qttClient.configureConnectDisconnectTimeout(1000)  # 1000 sec
        qttClient.connect()
        #qttClient = boto3.client('iot-data')
        qttClient.subscribe("AWS/Aarhus/traffic/alert", 0, customCallback)
        #app.run(debug=True, port=int(PORT))
        #app.run(debug=True, host='0.0.0.0', port=int(PORT))
        app.run(host='0.0.0.0', port=int(PORT))
    except:
        traceback.print_exc()