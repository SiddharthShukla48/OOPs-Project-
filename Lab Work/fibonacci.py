def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end = " ")
        a, b = b, a + b 

n = int(input("Enter the number of terms for the Fibonacci series: "))
fibonacci(n)

