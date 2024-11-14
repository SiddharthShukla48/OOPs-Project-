class Student:

#default constructor 
    def child(self):
        print("Inside child")

# non parameterised constructor
    def __init__(self):
        self.name = "abc"
        print(self.name)

# parameterised constructor

    def __init__(self, name = None, age= None):
        self.name = name
        self.age = age
        print(self.name)
        print(self.age)

        
kelly = Student('Kelly', 13)