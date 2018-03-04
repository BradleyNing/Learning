import paho.mqtt.publish as publish
import dht11
import RPi.GPIO as GPIO
import time

Temp_sensor=2
keepReporting = True

def RptData(instance):
    while keepReporting:
        result = instance.read()
        if result.is_valid():
            print 'temperature: '+str(result.temperature) + ', ' + \
                  'humidity: ' +str(result.humidity)
            publish.single('nzl_data/temperature', 
                            str(result.temperature), 
                            hostname='iot.eclipse.org')
            #publish.single('nzl_data/humidity', 
            #                str(result.humidity), 
            #                hostname='iot.eclipse.org')
        else :
            print 'error code: '+str(result.error_code)
        time.sleep(2.0)

def init():
  # Main program block
  GPIO.setwarnings(False)
  #GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
  GPIO.setmode(GPIO.BOARD)
  instance = dht11.DHT11(pin = Temp_sensor)
  return instance

if __name__ == '__main__':
    print 'Reporting data begining...'
    try:
        instance = init()
        RptData(instance)
    except KeyboardInterrupt:
        pass

    finally:
        GPIO.cleanup()