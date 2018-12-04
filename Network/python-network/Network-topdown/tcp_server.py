from socket import *

serverPort = 9999
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))

conNums = 1
serverSocket.listen(conNums)
print('The server is ready to receive')

while True:
	conSocket, addr = serverSocket.accept()
	msg = conSocket.recv(1024).decode()
	capMsg = msg.upper()
	conSocket.send(capMsg.encode())
	conSocket.close()
