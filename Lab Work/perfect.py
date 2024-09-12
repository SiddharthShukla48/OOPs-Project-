def isPerfect(num):
    if num < 1:
        return False  

    divisor_sum = 0
    for i in range(1, num):
        if num % i == 0:  
            divisor_sum += i

    return divisor_sum == num  

num = int(input("Enter a number to check if its perfect number or not -> "))
if isPerfect(num):
    print(f"{num} is a perfect number.")
else:
    print(f"{num} is not a perfect number.")
