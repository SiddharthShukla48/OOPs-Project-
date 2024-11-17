class BubbleSort:
    def __init__(self, data):
        self.data = data  # Initialize the class with the input data array.

    def sort(self):
        n = len(self.data) 
        for i in range(n):  
            for j in range(0, n - i - 1):  # Inner loop avoids already sorted elements.
                if self.data[j] > self.data[j + 1]: 
                    self.data[j], self.data[j + 1] = self.data[j + 1], self.data[j] 

    def get_sorted_data(self):
        return self.data 

data = [64, 34, 25, 12, 22, 11, 90]
bubble_sort = BubbleSort(data)  # Create an object with the input data.
bubble_sort.sort()  # Call the sort method to sort the data.
print(bubble_sort.get_sorted_data())  # Print the sorted data.
