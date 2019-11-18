
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
https://docs.python.org/3/library/logging.html
 
import logging

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
https://pythonworld.ru/novosti-mira-python/scientific-graphics-in-python.html

    # Arrows
    x_pos = [0, 1, 0.2]
    y_pos = [0, 1, 0.8]
    x_direct = [1, 0, 0.7]
    y_direct = [1, -1, -0.7]

    plt.quiver(x_pos, y_pos, x_direct, y_direct, scale=3)
    plt.show()

https://thispointer.com/pandas-dataframe-get-minimum-values-in-rows-or-columns-their-index-position/
https://askvoprosy.com/voprosy/pandas-subtract-row-mean-from-each-element-in-row
https://khashtamov.com/ru/pandas-introduction/
https://younglinux.info/python/feature/generators

temp_list = [(1, 3), (5, 4), (6, 2), (4, 1), (3, 6), (2, 5)]
d = dict(temp_list)
new_list = [min(temp_list)]
for _ in range(len(temp_list) - 1):
    x = new_list[-1][1]
    new_list.append((x, d.get(x)))

print(new_list)


# # Get a series containing minimum value of each column
# dj = df.min()
#
# print('minimum value in each column : ')
# print(dj)
#
# # Get a series containing minimum value of each row
# di = df.min(axis=1)
#
# print('minimum value in each row : ')
# print(di)

# aa = df.sub(di, axis=0)
# print(aa)

# bb = df.sub(dj, axis=1)
# print(bb)
#
# sum_vector = di.sum() + dj.sum()
# print(sum_vector)
#
# df.drop('2', axis=1, inplace=True)
# df.drop('4', axis=0, inplace=True)
# df = df.drop(df.columns[[1]], axis=1)
# print(df)
# print(df.iloc[1][1], df.loc['5']['4'])
