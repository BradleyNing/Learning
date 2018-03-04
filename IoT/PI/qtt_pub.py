import paho.mqtt.publish as publish
import time

print 'running...'
publish.single('AWS/test', 'Hello test', hostname='iot.eclipse.org')
while True:
	publish.single('AWS/test', 'Hello test', hostname='iot.eclipse.org')
	time.sleep(2)

