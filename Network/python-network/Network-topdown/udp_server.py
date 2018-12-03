from socket import *

serverPort = 9999
serverSocket = socket('', serverPort)
serverSocket.bind(('', serverPort))

print("The server is ready to receive")

while True:
	msg, clientAddr = serverSocket.recvfrom(2048)
	modifiedMsg = msg.decode().upper()
	serverSocket.sendto(modifiedMsg.encode(), clientAddr)

	

