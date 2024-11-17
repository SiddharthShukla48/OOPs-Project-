class InsertionSort:
    def __init__(self, data):
        self.data = data 

    def sort(self):
        for i in range(1, len(self.data)): 
            key = self.data[i]  # The element to be positioned correctly.
            j = i - 1
            while j >= 0 and key < self.data[j]:  # Compare key with each element to its left.
                self.data[j + 1] = self.data[j]  # Shift elements to the right to make space for the key.
                j -= 1
            self.data[j + 1] = key  # Place the key in its correct position.

    def get_sorted_data(self):
        return self.data  

data = [12, 11, 13, 5, 6]
insertion_sort = InsertionSort(data)  # Create an object with the input data.
insertion_sort.sort()  # Call the sort method to sort the data.
print(insertion_sort.get_sorted_data()) 
