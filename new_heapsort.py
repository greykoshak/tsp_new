import random
import time


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
        if len(self.heap) > 1:
            self.heap[0] = self.heap.pop()
        else:
            self.heap = []
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
arr = [random.randint(1, 100) for i in range(100_000)]

t = time.time()

# Heap creation
for i in range(0, len(arr)):
    h.add_element(arr[i])
# print("heap: ", heap)

# print("Проверка на дорогах-1...")
# fl = True
# for i in range(len(heap)):
#     if (2 * i + 1 < len(heap) and heap[i] < heap[2 * i + 1]) or (2 * i + 2 < len(heap) and heap[i] < heap[2 * i + 2]):
#         print("Error-1: {}, {}, {}".format(heap[i], heap[i + 1], heap[i + 22]))
#         fl = False
# print("Чисто") if fl else print("1 Не пройден!")

hs = HeapSort(h.get_heap())
sorted = hs.get_desc_sorted()

print("Время выполнения функции: {:5.2f}".format(time.time() - t))
print("Проверка на дорогах-3...")
fl = True
for i in range(len(sorted) - 1):
    if sorted[i] < sorted[i + 1]:
        print("Error-3: {}, {}".format(sorted[i], sorted[i + 1]))
        fl = False
print("Чисто") if fl else print("3 Не пройден!")

