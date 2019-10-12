
import time

def timer(f):
    def tmp(*args, **kwargs):
        t = time.time()
        res = f(*args, **kwargs)
        print("Время выполнения функции: {}".format(time.time() - t))
        return res

    return tmp


@timer
def func(x, y):
    return sqrt(x**2 + y**2)


res1 = func(3, 4)
print("res = {}".format(res1))


n = max(matrix, key=lambda x:x[1]) # Максимальное значение по второму элементу