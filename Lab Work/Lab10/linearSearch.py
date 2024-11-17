def linear_search(arr, target):
    for index, value in enumerate(arr): 
        if value == target: 
            return index  
    return -1  

arr = [2, 4, 6, 8, 10]
target = 6
print(linear_search(arr, target)) 
