def mul_table(num, upto):
    for i in range(1, upto + 1):
        print(f"{num} x {i} = {num * i}")

# Example usage
num = int(input("Enter a number for its multiplication table -> "))
upto = int(input("Enter the range -> "))
mul_table(num, upto)
