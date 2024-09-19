def compute_power(x, n):
    return x ** n  

x = float(input("Enter the value of X: "))  
n = int(input("Enter the value of n: "))  

result = compute_power(x, n)

print(f"{x} raised to the power of {n} is {result}")
