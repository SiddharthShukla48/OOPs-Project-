import math

class ComplexNumber:
    def __init__(self, real, imag):
        """Initialize the complex number with real and imaginary parts."""
        self.real = real
        self.imag = imag

    def add(self, c):
        """Add another complex number and return a new ComplexNumber instance."""
        return ComplexNumber(self.real + c.real, self.imag + c.imag)

    def magnitude(self):
        """Calculate and return the magnitude of the complex number."""
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        """String representation of the complex number in a+bi format."""
        return f"{self.real} + {self.imag}i"


# Create two ComplexNumber objects
c1 = ComplexNumber(3, 4)
c2 = ComplexNumber(1, 2)

# Perform addition
sum_result = c1.add(c2)

# Print results
print("First Complex Number:", c1)
print("Second Complex Number:", c2)
print("Sum:", sum_result)
print("Magnitude of the first complex number:", c1.magnitude())
print("Magnitude of the second complex number:", c2.magnitude())
print("Magnitude of the sum:", sum_result.magnitude())
