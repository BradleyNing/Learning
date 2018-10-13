class Dog(object):
	species = 'mamal'

	def __init__(self, breed, name):
		self.breed = breed
		self.name = name

class Circle():
	pi = 3.14
	def __init__(self, radius=1):
		self.radius = radius

	def area(self):
		return self.radius**2 * Circle.pi

mydog = Dog('Lab', 'Sammy')
print(mydog.name)
print(mydog.species)

Dog.species = 'test'
print(mydog.species)

myc = Circle(3)
print(myc.area())
print(myc.pi)
