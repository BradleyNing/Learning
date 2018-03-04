from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, IntegerField
from wtforms import validators

class Form_admin_get(FlaskForm):
    getID = IntegerField('Input the Sensor ID, ID=-1 for retrieve all IDs which are currently alerted',
                        validators=[validators.Required()])
    submitGet = SubmitField('Confirm')
    getInfoText = TextAreaField('Information of the ID or latest alerts')
    
class Form_admin_put(FlaskForm):
    putID = IntegerField('Input the Sensor ID, ID=-1 for update all sensors',
                        validators=[validators.Required()])
    inputSpeedGate = IntegerField('Input the Vihecle Speed Gate Value(Integer) to be Alerted',
                        validators=[validators.Required()])
    inputCountGate = IntegerField('Input the Vihecle Count Gate Value(Integer) to be Alerted',
                        validators=[validators.Required()])
    submitPut = SubmitField('Confirm')
    putInfoText = TextAreaField('Information of the Update')

class Form_admin_show(FlaskForm):   
    showAllSensors = SubmitField('Show all sensors on Map')
    showAlertedSensors = SubmitField('Show alerted sensors on Map')
    showTogether = SubmitField('Show alerted and normal sensors together')

class Form_led_config(FlaskForm):
    ledAlarmGate = IntegerField('Input how many alerted sensors number to Blink RED LED',
                        default=10)
    submitRedLed = SubmitField('Red LED Blinking')
    submitGreenLed = SubmitField('Green LED Blinking')
