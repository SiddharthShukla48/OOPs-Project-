def is_harshad(num):
    str_num = str(num)
    digit_sum = sum(int(digit) for digit in str_num)

    return num % digit_sum == 0

number = int(input("Enter a number to check if it is Harshad number or not : "))

if is_harshad(number):
    print(f"{number} is a Harshad number.")
else:
    print(f"{number} is not a Harshad number.")
