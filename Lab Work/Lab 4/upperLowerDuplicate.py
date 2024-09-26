sequence = "HelloWorld"
unique_chars = set(sequence)

uppercase_result = ''.join(map(str.upper, unique_chars))
lowercase_result = ''.join(map(str.lower, unique_chars))

print("Uppercase:", uppercase_result)
print("Lowercase:", lowercase_result)
