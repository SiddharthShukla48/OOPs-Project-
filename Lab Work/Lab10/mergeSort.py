class MergeSort:
    def __init__(self, data):
        self.data = data  

    def sort(self):
        self.data = self._merge_sort(self.data)  # Start the merge sort process.

    def _merge_sort(self, arr):
        if len(arr) <= 1:  # Base case: a list with one element is already sorted.
            return arr
        
        # Split the array into two halves.
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursively sort both halves and merge them.
        left_sorted = self._merge_sort(left_half)
        right_sorted = self._merge_sort(right_half)

        # Merge the sorted halves and return.
        return self._merge(left_sorted, right_sorted)

    def _merge(self, left, right):
        merged = []
        i = j = 0

        # Merge two sorted lists into one.
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        
        # If there are any remaining elements in either list, append them.
        merged.extend(left[i:])
        merged.extend(right[j:])
        
        return merged

    def get_sorted_data(self):
        return self.data 

data = [38, 27, 43, 3, 9, 82, 10]
merge_sort = MergeSort(data)  # Create an object with the input data.
merge_sort.sort()  # Call the sort method to sort the data.
print(merge_sort.get_sorted_data())  # Print the sorted data.
