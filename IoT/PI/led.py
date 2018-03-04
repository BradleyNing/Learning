import RPi.GPIO as GPIO
import time

led=40
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led, GPIO.OUT)
GPIO.output(led, GPIO.LOW)
while True:
	GPIO.output(led, GPIO.HIGH)
	time.sleep(0.5)
	GPIO.output(led, GPIO.LOW)
	time.sleep(0.5)

