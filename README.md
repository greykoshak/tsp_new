
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

# https://python-scripts.com/logging-python

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    # create the logging file handler
    fh = logging.FileHandler("raw_tsp.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(fh)

    logger.info("Program started")

https://www.python-course.eu/python3_lambda.php
https://ru.stackoverflow.com/questions/1037560/%d0%a1%d0%be%d1%80%d1%82%d0%b8%d1%80%d0%be%d0%b2%d0%ba%d0%b0-%d1%81%d0%bf%d0%b8%d1%81%d0%ba%d0%b0-%d0%ba%d0%be%d1%80%d1%82%d0%b5%d0%b6%d0%b5%d0%b9

