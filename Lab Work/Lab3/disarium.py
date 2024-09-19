def is_disarium(num):
   
    str_num = str(num)
    length = len(str_num)
    
    disarium_sum = 0  
    for i in range(length):
        disarium_sum += int(str_num[i]) ** (i + 1)
    
    return disarium_sum == num

number = int(input("Enter a number to check if its disarium or not : "))

if is_disarium(number):
    print(f"{number} is a Disarium number.")
else:
    print(f"{number} is not a Disarium number.")
