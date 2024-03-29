import csv
import numpy as np
import pandas as pd

import heapq

def get_highest_with_index1(arr, n):
    # Use heapq's nlargest to get the n highest elements
    highest_values = heapq.nlargest(n, arr)
    
    # Initialize a 2D array to store both the number and its index
    result = []
    
    # Iterate over the highest values to find their indices and store them in the result array
    for value in highest_values:
        index = arr.index(value)
        result.append([value, index])
    
    return result

# # Example usage:
# arr = [4, 8, 2, 6, 10, 3, 7, 1, 5, 9]
# n = 10

# highest_with_index = get_highest_with_index(arr, n)
# print(highest_with_index)


def get_highest_with_index(df, n):
    # Get the n largest items from the DataFrame
    highest_values = df.nlargest(n, 'Values',"all")
    
    # Create a 2D array to store the number and its index
    result = []
    
    # Iterate over the highest values to find their indices and store them in the result array
    for index, row in highest_values.iterrows():
        result.append([row['Values'], index])
    
    return result

# import numpy as np

# # Example 2D NumPy array
# array_2d = np.array([[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]])

# # Extracting the second values from each row
# second_values = array_2d[:, 1]

# print(second_values)



if __name__ == "__main__":

    data = {'Values': [4, 8, 2, 6, 10, 3, 7, 1, 5, 9]}
    df = pd.DataFrame(data)

    # Example usage:
    arr = [4, 8, 2, 6, 10, 3, 7, 1, 5, 9]
    n = 10

    highest_with_index = get_highest_with_index(df, n)
    print(highest_with_index)
    print(df.head)
    second_values = highest_with_index[:, 1]
    print(second_values)

    # ###########################################################################
    # # Define the path to your CSV file
    # csv_file_path = 'keras_yamnet\yamnet_class_map.csv'

    # # Initialize an empty list to store the data
    # data = []



    # # Open the CSV file and read its contents
    # with open(csv_file_path, newline='') as csvfile:
    #     csv_reader = csv.reader(csvfile)
    #     # Skip the header row if needed
    #     next(csv_reader, None)
    #     # Iterate over each row in the CSV file
    #     for row in csv_reader:
    #         # Append the row to the data list
    #         data.append(row)

    # # Convert the list to a NumPy array
    # data_array = np.array(data)

    # # Print the array
    # print(data_array)
    # for i in data_array:
    #     print(i[2])

