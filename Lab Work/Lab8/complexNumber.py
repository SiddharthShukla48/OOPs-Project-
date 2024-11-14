import math

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def add(self, c):
        return ComplexNumber(self.real + c.real, self.imag + c.imag)

    def magnitude(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        return f"{self.real} + {self.imag}i"


c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)

sum_result = c1.add(c2)

print("First Complex Number:", c1)
print("Second Complex Number:", c2)
print("Sum:", sum_result)
print("Magnitude of the first complex number:", c1.magnitude())
print("Magnitude of the second complex number:", c2.magnitude())
print("Magnitude of the sum:", sum_result.magnitude())
