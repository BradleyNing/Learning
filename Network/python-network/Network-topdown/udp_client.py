from socket import *

serverName = '192.168.1.152'
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_DGRAM)

msg = input('Input lowercase sentense: ')
clientSocket.sendto(msg.encode(), (serverName, serverPort))

modfiedMsg, serverAddress = clientSocket.recvfrom(2048)
print(modfiedMsg.decode())
clientSocket.close()