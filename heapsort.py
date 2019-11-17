# Python program for implementation of heap Sort
import random
import time


def timer(f):
    def tmp(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        print("Время выполнения функции: {}".format(time.time() - t))
        return res

    return tmp


# To heapify subtree rooted at index i. n is size of heap
def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is greater than root
    # if l < n and arr[i] < arr[l]:
    if l < n and arr[i] > arr[l]:
        largest = l

    # See if right child of root exists and is greater than root
    # if r < n and arr[largest] < arr[r]:
    if r < n and arr[largest] > arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)


# The main function to sort an array of given size
@timer
def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


# Driver code to test above
arr = [random.randint(1, 100_000) for i in range(1_000_002)]
heapSort(arr)
n = len(arr)
# print("Sorted array is")
# for i in range(n):
#     print("%d" % arr[i]),
# This code is contributed by Mohit Kumra

print("Проверка на дорогах-3...")
fl = True
for i in range(len(arr) - 1):
    if arr[i] < arr[i + 1]:
        print("Error-3: {}, {}".format(arr[i], arr[i + 1]))
        fl = False
print("Чисто") if fl else print("3 Не пройден!")
