class Student:
# constructor
    def __init__(self, name):
        print("Inside Constructor")
        self.name = name
        print("Object Initialised")
    def show(self):
        print('Hello, my name is', self.name)

# destructor
    def __del__(self):
        print("Inside Destructor")
        print("Object Destroyed")

# create object

s1 = Student("abc")
s1.show()

# delete object
del s1