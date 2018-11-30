import requests
#, 
def geocode(address):
	parameters = {'address': address, 'sensor': 'false', \
				  'key':'AIzaSyCWrr1EuW5xjvyYvzpmp1LLyjUX7rkqfCE'}
	base = 'https://maps.googleapis.com/maps/api/geocode/json'
	response = requests.get(base, params=parameters)
	answer = response.json()
	print(answer['results'][0]['geometry']['location'])

if __name__ == '__main__':
	geocode('Mountain View, CA 94043, USA')