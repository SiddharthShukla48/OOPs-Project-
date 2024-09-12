def gcd(a, b):
    while b:
        a, b = b, a % b  
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)  

num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))

gcd_val = gcd(num1, num2)
lcm_val = lcm(num1, num2)

print(f"GCD of {num1} and {num2} is: {gcd_val}")
print(f"LCM of {num1} and {num2} is: {lcm_val}")
