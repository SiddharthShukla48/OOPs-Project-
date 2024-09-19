def isArmstrong(num):
    str_num = str(num)  
    num_digits = len(str_num)  

    power_sum = 0  #

    for digit in str_num:
        power_sum += int(digit) ** num_digits

    return power_sum == num 

print("Armstrong numbers from 1 to 1000 are:")
for num in range(1, 1001):
    if isArmstrong(num):  
        print(num)
