def count_case(s):
    uppercase_count = sum(1 for char in s if char.isupper())
    lowercase_count = sum(1 for char in s if char.islower())
    return uppercase_count, lowercase_count

input_string = input("Enter a string: ")

upper_count, lower_count = count_case(input_string)

print(f"Uppercase Letters: {upper_count}")
print(f"Lowercase Letters: {lower_count}")
