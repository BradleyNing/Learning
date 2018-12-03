from socket import *

serverPort = 9999
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', serverPort))

print("The server is ready to receive")

while True:
	msg, clientAddr = serverSocket.recvfrom(2048)
	modifiedMsg = msg.decode().upper()
	serverSocket.sendto(modifiedMsg.encode(), clientAddr)



