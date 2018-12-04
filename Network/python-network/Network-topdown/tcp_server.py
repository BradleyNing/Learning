from socket import *

serverPort = 9999
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))

conNums = 1
serverSocket.listen(conNums)
print('The server is ready to receive')

while True:
	conSocket, clientAddr = serverSocket.accept()
	print("received connection from {} and setup socket of {}" \
		   .format(str(clientAddr), str(conSocket)))
	msg = conSocket.recv(1024).decode()
	print("received message from {}: {}".format(str(clientAddr), msg))
	capMsg = msg.upper()
	conSocket.send(capMsg.encode())
	conSocket.close()
