import socket

if __name__ == '__main__':
	hostname = 'google.co.uk'
	addr = socket.gethostbyname(hostname)
	print('The IP address of {} is {}'.format(hostname, addr))
	service = 'ssh'
	print(service+"'s port is " + str(socket.getservbyname('ssh')))

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	data = 'test'
	sock.sendto(data, ('127.0.0.1', 8000))
	print('The OS assigned me the address {}'.format(sock.getsockname()))
