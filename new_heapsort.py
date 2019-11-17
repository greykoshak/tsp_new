import random
import time


def add_element(ptr_heap, x):
    ptr_heap.append(x)
    p = len(ptr_heap) - 1
    fl_moved = True

    while fl_moved and p > 0:
        fl_moved = False
        if ptr_heap[p] > ptr_heap[(p - 1) // 2]:
            ptr_heap[p], ptr_heap[(p - 1) // 2] = ptr_heap[(p - 1) // 2], ptr_heap[p]
            p = (p - 1) // 2
            fl_moved = True


def del_element(ptr_heap):
    ptr_heap[0] = ptr_heap.pop()
    n = len(ptr_heap)
    p = 0
    fl_moved = True

    while fl_moved and (p * 2 + 1) < n:
        fl_moved = False
        left = p * 2 + 1
        right = left + 1
        ptr_max = left
        if (right < n) and ptr_heap[right] > ptr_heap[left]:
            ptr_max = right
        if ptr_heap[p] < ptr_heap[ptr_max]:
            ptr_heap[p], ptr_heap[ptr_max] = ptr_heap[ptr_max], ptr_heap[p]
            p = ptr_max
            fl_moved = True


heap = []
arr = [random.randint(1, 100_000) for i in range(1_000_002)]

t = time.time()

# Heap creation
for i in range(0, len(arr)):
    add_element(heap, arr[i])
# print("heap: ", heap)

# print("Проверка на дорогах-1...")
# fl = True
# for i in range(len(heap)):
#     if (2 * i + 1 < len(heap) and heap[i] < heap[2 * i + 1]) or (2 * i + 2 < len(heap) and heap[i] < heap[2 * i + 2]):
#         print("Error-1: {}, {}, {}".format(heap[i], heap[i + 1], heap[i + 22]))
#         fl = False
# print("Чисто") if fl else print("1 Не пройден!")

sorted = []
for i in range(len(heap) - 1, 0, -1):
    sorted.append(heap[0])
    del_element(heap)
print("Время выполнения функции: {:5.2f}".format(time.time() - t))
# print("sorted: ", sorted, "\n", sorted[::-1])

print("Проверка на дорогах-3...")
fl = True
for i in range(len(sorted) - 1):
    if sorted[i] < sorted[i + 1]:
        print("Error-3: {}, {}".format(sorted[i], sorted[i + 1]))
        fl = False
print("Чисто") if fl else print("3 Не пройден!")
