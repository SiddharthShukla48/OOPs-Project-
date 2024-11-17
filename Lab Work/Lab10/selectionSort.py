class SelectionSort:
    def __init__(self, data):
        self.data = data 

    def sort(self):
        n = len(self.data) 
        for i in range(n): 
            min_idx = i  
            for j in range(i + 1, n):  
                if self.data[j] < self.data[min_idx]: 
                    min_idx = j
            self.data[i], self.data[min_idx] = self.data[min_idx], self.data[i]  

    def get_sorted_data(self):
        return self.data  

data = [64, 25, 12, 22, 11]
selection_sort = SelectionSort(data)  # Create an object with the input data.
selection_sort.sort()  # Call the sort method to sort the data.
print(selection_sort.get_sorted_data())  # Print the sorted data.
