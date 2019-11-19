import random


class Heapify:
    """ Создание кучи, сортировка: методы """

    def __init__(self, ptr_heap=None):
        self.heap = list() if ptr_heap is None else ptr_heap

    def add_element(self, x):
        """ Добавить элемент в кучу """
        self.heap.append(x)
        p = len(self.heap) - 1
        fl_moved = True

        while fl_moved and p > 0:
            fl_moved = False
            if self.heap[p] > self.heap[(p - 1) // 2]:
                self.heap[p], self.heap[(p - 1) // 2] = self.heap[(p - 1) // 2], self.heap[p]
                p = (p - 1) // 2
                fl_moved = True

    def del_element(self):
        """ Перестроить кучу после удаления корня дерева """
        node = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = node
        n = len(self.heap)
        p = 0
        fl_moved = True

        while fl_moved and (p * 2 + 1) < n:
            fl_moved = False
            left = p * 2 + 1
            right = left + 1
            ptr_max = left
            if (right < n) and self.heap[right] > self.heap[left]:
                ptr_max = right
            if self.heap[p] < self.heap[ptr_max]:
                self.heap[p], self.heap[ptr_max] = self.heap[ptr_max], self.heap[p]
                p = ptr_max
                fl_moved = True

    def get_heap(self):
        return self.heap


class HeapSort(Heapify):
    """ Сортировка на базе кучм """

    def __init__(self, ptr_heap):
        self.heap = ptr_heap[:]
        self._sorted = list()

    def get_desc_sorted(self):
        for i in range(len(self.heap) - 1, -1, -1):
            self._sorted.append(self.heap[0])
            self.del_element()
        return self._sorted


h = Heapify()
arr = [random.randint(1, 100) for i in range(17)]

# Heap creation
for i in range(0, len(arr)):
    h.add_element(arr[i])
# print("heap: ", heap)

# print("Проверка на дорогах-1...")
# fl = True
# for i in range(len(h.heap)):
#     if (2 * i + 1 < len(h.heap) and h.heap[i] < h.heap[2 * i + 1]) or (
#                     2 * i + 2 < len(h.heap) and h.heap[i] < h.heap[2 * i + 2]):
#         print("Error-1: {}, {}, {}".format(h.heap[i], h.heap[i + 1], h.heap[i + 22]))
#         fl = False
# print("Чисто") if fl else print("1 Не пройден!")

hs = HeapSort(h.get_heap())
arr_sorted = hs.get_desc_sorted()
print(arr, "\n", arr_sorted)

print("Проверка на дорогах-3...")
fl = True
for i in range(len(arr_sorted) - 1):
    if arr_sorted[i] < arr_sorted[i + 1]:
        print("Error-3: {}, {}".format(arr_sorted[i], arr_sorted[i + 1]))
        fl = False
print("Чисто") if fl else print("3 Не пройден!")
