import math
import numpy as np 

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return sum((b - a) ** 2 for a, b in zip(x1, x2)) ** 0.5

# Example lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]

# Calculate and print the distance
x = euclidean_distance(list1, list2)
print("Euclidean distance between two lists is:", x)
