my_list = [5, 2, 9, 1, 5, 6]

my_list.append(7)
print("After append(7):", my_list)

my_list.insert(2, 3)  
print("After insert(2, 3):", my_list)


my_list.remove(5)  
print("After remove(5):", my_list)

popped_element = my_list.pop()  
print("After pop():", my_list)
print("Popped element:", popped_element)

my_list.sort()
print("After sort():", my_list)

my_list.reverse()
print("After reverse():", my_list)

count_of_3 = my_list.count(3)
print("Count of 3:", count_of_3)

index_of_9 = my_list.index(9)
print("Index of 9:", index_of_9)

my_list.extend([8, 4, 2])
print("After extend([8, 4, 2]):", my_list)

my_list.clear()
print("After clear():", my_list)
