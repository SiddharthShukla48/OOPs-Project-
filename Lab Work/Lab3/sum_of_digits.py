def sum_of_digits(num):
    
    digit_sum = sum(int(digit) for digit in str(num))
    return digit_sum

num = int(input("Enter a number: "))

result = sum_of_digits(num)

print(f"The sum of the digits in {num} is {result}.")
