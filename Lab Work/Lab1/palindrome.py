def is_palindrome(num):
    return str(num) == str(num)[::-1]

num = int(input("Enter a number to check if it's a palindrome number or not ->  "))

if is_palindrome(num):
    print(f"{num} is a palindrome number.")
else:
    print(f"{num} is not a palindrome number.")

