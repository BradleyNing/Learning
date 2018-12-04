from socket import *

hp = '192.168.1.193'
mac = '192.168.1.152'
serveName = mac
serverPort = 9999

clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((serveName, serverPort))

msg = input('Input a lowercase sentence: ')
clientSocket.send(msg.encode())

modifiedMsg = clientSocket.recv(1024)
print('Received from server: ', modifiedMsg.decode())
clientSocket.close()

