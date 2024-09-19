def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def isStrong_number(num):
    str_num = str(num) 
    fact_sum = sum(factorial(int(digit)) for digit in str_num) 
    
    return fact_sum == num  

num = int(input("Enter a number to check if its strong number or not -> "))
if isStrong_number(num):
    print(f"{num} is a strong number.")
else:
    print(f"{num} is not a strong number.")
