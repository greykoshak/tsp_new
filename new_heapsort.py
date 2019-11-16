def add_element(heap, x):
    heap.append(x)
    p = len(heap) - 1

    while p > 0 and heap[p] > heap[p // 2]:
        heap[p], heap[p // 2] = heap[p // 2], heap[p]
        p //= 2


def del_element(ptr_heap, heap_len):
    ptr_heap[heap_len], ptr_heap[0] = ptr_heap[0], ptr_heap[heap_len]
    n = heap_len - 1
    p = 0
    moved = True

    while moved and p * 2 <= n:
        moved = False
        max = p * 2 + 1
        if (max + 1) and ptr_heap[max + 1] > ptr_heap[max]:
            max += 1
        if ptr_heap[p] < ptr_heap[max]:
            ptr_heap[p], ptr_heap[max] = ptr_heap[max], ptr_heap[p]
            p = max
            moved = True


heap = []
arr = [12, 11, 13, 5, 6, 7, 3, 7, 14, 1, 4, 1, 73, 4, 17, 3, 21, 81]

# Heap creation
for i in range(0, len(arr)):
    add_element(heap, arr[i])
print("heap: ", heap)
for i in range(len(heap)-1, -1, -1):
    del_element(heap, i)
print("sorted: ", heap)
