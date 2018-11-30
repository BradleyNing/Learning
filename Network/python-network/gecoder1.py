# google maps api key AIzaSyCWrr1EuW5xjvyYvzpmp1LLyjUX7rkqfCE 
from pygeocoder import Geocoder

if __name__ == '__main__':
	address = 'Mountain View, CA'
	print(Geocoder.geocode(address)[0].coordinates)