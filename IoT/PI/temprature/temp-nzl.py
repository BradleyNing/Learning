import dht11
import RPi.GPIO as GPIO
import time
Temp_sensor=14
Temp_sensor=12

def main():
  # Main program block
  GPIO.setwarnings(False)
  GPIO.setmode(GPIO.BOARD)       # Use BCM GPIO numbers
  instance = dht11.DHT11(pin = Temp_sensor)

  while True:
    #get DHT11 sensor value
    result = instance.read()
    #print result
    if result.is_valid():
        print("Temperature: %d C" % result.temperature)
        print("Humidity: %d %%" % result.humidity)
    else :
        print("Error: %d" % result.error_code)
    time.sleep(2.0)
    # Send some test

if __name__ == '__main__':

  try:
    main()
  except KeyboardInterrupt:
    pass
  finally:
    GPIO.cleanup()