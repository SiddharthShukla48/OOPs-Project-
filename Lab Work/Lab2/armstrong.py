def isArmstrong(num):
   
    str_num = str(num)
    num_digits = len(str_num) 

    power_sum = 0

    for digit in str_num:
        power_sum += int(digit) ** num_digits  

    if power_sum == num:
        return True
    else:
        return False

num = int(input("Enter a number to check if its armstrong or not -> "))
if isArmstrong(num):
    print(f"{num} is an Armstrong number.")
else:
    print(f"{num} is not an Armstrong number.")
