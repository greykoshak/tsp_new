import random


def add_element(heap, x):
    heap.append(x)
    p = len(heap) - 1

    while p > 0 and heap[p] > heap[p // 2]:
        heap[p], heap[p // 2] = heap[p // 2], heap[p]
        p //= 2


def del_element(ptr_heap):
    ptr_heap[0] = ptr_heap.pop()
    n = len(ptr_heap) - 1
    p = 0
    moved = True

    while moved and (p * 2 + 1) <= n:
        moved = False
        max = p * 2 + 1
        if (max + 1 <= n) and ptr_heap[max + 1] > ptr_heap[max]:
            max += 1
        if ptr_heap[p] < ptr_heap[max]:
            ptr_heap[p], ptr_heap[max] = ptr_heap[max], ptr_heap[p]
            p = max
            moved = True


heap = []
# arr = [12, 11, 13, 5, 6, 7, 3, 7, 14, 1, 4, 1, 73, 4, 17, 3, 21, 81]
arr = [random.randint(1, 100) for i in range(150)]

# Heap creation
for i in range(0, len(arr)):
    add_element(heap, arr[i])
print("heap: ", heap)

sorted = []
for i in range(len(heap) - 1, 0, -1):
    sorted.append(heap[0])
    del_element(heap)
print("sorted: ", sorted, "\n", sorted[::-1])

print("Проверка на дорогах...")
fl = True
for i in range(len(sorted) - 1):
    if sorted[i] < sorted[i + 1]:
        print("Error: {}, {}".format(sorted[i], sorted[i + 1]))
        fl = False
print("Чисто") if fl else print("Не пройден!")

