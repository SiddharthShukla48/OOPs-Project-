from functools import reduce

n = int(input("Enter N : "))
num = range(1,n+1)
result = reduce(lambda x,y:x*y,num)
print(f"Factorial of N is {result}")