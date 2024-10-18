class Student:

#default constructor 
    def child(self):
        print("Inside child")

# non parameterised constructor
    def __init__(self):
        self.name = "abc"

# non parameterised constructor
    def __init__(self):
        self.name = "abc"
        print(self.name)

# parameterised constructor

    def __init__(self, name, age):
        self.name = name
        self.age = age

        
# creating first object
emma = Student()
# creating Second object
kelly = Student('Kelly', 13)