import math  

def compute_nCr(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

n = int(input("Enter the value of n: "))
r = int(input("Enter the value of r: "))

if r > n or n < 0 or r < 0:
    print("Invalid input: r should be less than or equal to n and both n and r should be non-negative.")
else:
    result = compute_nCr(n, r)
    
    print(f"{n}C{r} = {result}")

