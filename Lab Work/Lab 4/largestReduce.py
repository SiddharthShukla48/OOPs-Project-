from functools import reduce

p = int(input("Enter p = "))
q = int(input("Enter q = "))
r = int(input("Enter r = "))

result = reduce(lambda x,y: x if x > y else y, [p,q,r])
print(f"Largest number is {result}")