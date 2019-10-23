# Travelling salesman problem
import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt

points = [(800, 400), (500, 300), (900, 500), (900, 400), (700, 100), (500, 500), (900, 300), (100, 300)]


class DefineMatrix:
    """ Построить матрицу расстояний между точками по их координатам """

    def __init__(self, coord: list):
        self.points = coord

    def build_matrix(self):
        n = len(self.points)  # Размерность матрицы
        cities = [City(i, point[0], point[1]) for i, point in enumerate(self.points)]
        distances = np.zeros((n, n), dtype=float)

        for i in np.arange(0, n, 1):
            for j in np.arange(i, n, 1):
                if i == j:
                    distances[i, j] = float('inf')
                    continue
                distances[i, j] = City.distance(cities[i], cities[j])
                distances[j][i] = distances[i][j]
        return distances


class City:
    """ Информация о городе """

    def __init__(self, number, x, y):
        self.number = number
        self.x = x
        self.y = y

    @staticmethod
    def distance(c1, c2):
        # distance = np.sqrt(np.sum((a - b) ** 2))
        return sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    def get_number_city(self):
        return self.number


class SetMatrix:
    """ Использовать заданную матрицу расстояний """

    def __init__(self, _matr=None):
        self.mat = np.array([[-1., 31., 15., 19., 8., 55],
                             [19., -1, 22., 31., 7., 35],
                             [25., 43., -1., 53., 57., 16],
                             [5., 50., 49., -1., 39., 9],
                             [24., 24., 33., 5., -1., 14],
                             [34., 26., 6., 3., 36., -1.]]) if _matr is None else _matr
        # self.mat = np.array([[-1., 10., 25., 25., 10.],
        #                      [1., -1, 10., 15., 2.],
        #                      [8., 9., -1., 20., 10.],
        #                      [14., 10., 24., -1., 15.],
        #                      [10., 8., 25., 27., -1.]])
        self.index = [[j for j in range(self.mat.shape[0])] for _ in range(2)]  # Индексы строк и столбцов
        self.set_diagonal()

    def set_diagonal(self):
        """ Диагональные элементы исключаем """
        diag = np.diag_indices(self.mat.shape[0])
        self.mat[diag] = float('inf')

    def get_matrix(self):
        return self.mat

    def get_index(self):
        return self.index

    def set_index(self, update_list):
        self.index = update_list

    def mod_index(self, new_edge: tuple, my_index=None) -> tuple:
        if my_index is None:
            func = lambda x: (self.index[0].index(x[0]), self.index[1].index(x[1]))
        else:
            func = lambda x: (my_index[0].index(x[0]), my_index[1].index(x[1]))
        return func(new_edge)


class GraphScore:
    """ Методы поиска в алгоритме ветвей и границ """

    def __init__(self, matrix):
        self.mat = matrix
        self.f0 = 0.
        self.final = 0.
        self.f0_root = list()

        self.f0_estimate()

    def f0_estimate(self):
        """ Первичная оценка нулевого варианта F0=mat(0,1)+mat(1,2)+mat(2,3)+mat(3,4)+mat(4,0)=10+10+20+15+10=65 """

        for _i in range(self.mat.shape[0] - 1):
            self.f0 += self.mat[_i][_i + 1]
            self.f0_root.append((_i, _i + 1))
        self.f0 += self.mat[self.mat.shape[0] - 1][0]  # f0 = 65
        self.f0_root.append((self.mat.shape[0] - 1, 0))

    def get_estimation(self):
        return self.f0_root, self.f0

    def get_root_estimation(self, my_root):
        """ Оценка заданного варианта """

        for pnt in my_root:
            self.final += self.mat[pnt[0]][pnt[1]]
        return self.final


def count_d(matrix):
    """ d - сумма di + dj (сумма минимальных элементов по строкам и столбцам) """

    dim = matrix.shape[0]  # Количество точек обхода
    di = np.zeros((dim, 1))  # Одномерный vector для выбора минимального значения по строке
    dj = np.zeros((1, dim))  # Одномерный vector для выбора минимального значения по столбцу

    di = np.min(matrix, axis=1)  # min элемент по строкам
    di.shape = (dim, 1)  # Преобразование вектора-строки в вектор-столбец
    matrix = matrix - di  # Редукция строк

    dj = np.min(matrix, axis=0)  # min элемент по столбцам
    matrix = matrix - dj  # Редукция столбцов
    return di.sum() + dj.sum(), matrix


def graph_edge(p_matrix, p_ind):
    """ Оценка нулевых элементов для поиска ребра графа -кандидата на включение в маршрут """

    _find = np.where(p_matrix == 0)  # Найти все нулевые элементы
    _v_null = zip(_find[0], _find[1])  # Вектор, содержащий координаты нулевых элементов
    max_value = list()  # Оценки нулевых точек
    pnt = list()  # Координаты нулевых точек
    inf = float('inf')

    for k in _v_null:
        p_matrix[k[0], k[1]], inf = inf, p_matrix[k[0], k[1]]
        di = mat[k[0]:k[0] + 1, ].min()  # min по строке
        dj = mat[:, k[1]].min()  # min по столбцу
        max_value.append(di + dj)
        pnt.append((p_ind[0][k[0]], p_ind[1][k[1]]))
        p_matrix[k[0], k[1]], inf = inf, p_matrix[k[0], k[1]]
    idx = max_value.index(max(max_value))
    return pnt[idx]  # Ребро с реальными узлами


def select_wrong_root(my_root: list, new_edge: tuple) -> list:
    # Строим максимальную цепочку от нового кандидата edge
    path = [new_edge]  # [(3, 1)]
    left_hand, right_hand = new_edge[0], new_edge[1]

    for _ in range(len(my_root)):
        for xy in my_root:
            if right_hand == xy[0]:
                path.append(xy)
                right_hand = xy[1]
            elif left_hand == xy[1]:
                path.insert(0, xy)
                left_hand = xy[0]
    wrong_root = [(new_edge[1], new_edge[0]), (right_hand, left_hand)]
    return wrong_root


def sort_root(my_root: list) -> list:
    """ Создать цепочку последовательных звеньев """

    path = [min(my_root)]
    for _ in range(len(my_root) - 1):
        next_item = path[-1]
        path.append(
            next(filter(lambda x: next_item[1] == x[0], my_root))
        )
    return path


if __name__ == "__main__":
    mat = DefineMatrix(points).build_matrix()

    sm = SetMatrix(mat)
    mat = sm.get_matrix()

    plans = list()  # Планы
    est_plans = list()  # Оценка планов
    root = list()  # Маршрут комивояжера

    igs = GraphScore(mat)
    print(igs.f0_root, igs.f0)
    first_pass = True
    build_root = True  # Условие работы цикла: пока есть ненулевые элементы

    while build_root:
        if first_pass:
            d_tuple = count_d(mat)
            d_min, mat = d_tuple[0], d_tuple[1]  # Оценка минимума минимумов =58 и новая матрица
            est_plans.append(d_min)

            if d_min == igs.f0:
                root.append(igs.f0_root)
                break
            first_pass = False
        else:
            ind = sm.get_index()
            edge = graph_edge(mat, ind)  # Поиск ребра-кандидата графа (реальные узлы)

            # Вариант "вправо" - считаем, что ребро edge не входит в маршрут
            right = mat.copy()
            ii = sm.mod_index(edge)
            right[ii] = float('inf')  # Исключаем ребро из маршрута

            d_tuple_right = count_d(right)
            d_right = est_plans[-1] + d_tuple_right[0]
            right = d_tuple_right[1]
            # print("d_right: ", d_tuple_right[0])

            # Вариант "влево" - считаем, что ребро edge входит в маршрут
            left = mat.copy()

            # left_ind = ind.copy()  # Не работает:
            # If the list contains objects and you want to copy them as well, use generic copy.deepcopy()
            left_ind = copy.deepcopy(ind)
            left = np.delete(left, (ii[0]), axis=0)
            left = np.delete(left, (ii[1]), axis=1)
            del left_ind[0][ii[0]]
            del left_ind[1][ii[1]]

            inf_list = select_wrong_root(root, edge)  # Исключаем ребра, образующие цикл с уже существующим root

            f = lambda x, ind_list: x in ind_list  # Проверка точки(x,y) на принадлежность текущей матрице

            for p in inf_list:
                if f(p[0], left_ind[0]) and f(p[1], left_ind[1]):
                    kk = sm.mod_index(p, left_ind)
                    left[kk] = float('inf')

            d_tuple_left = count_d(left)
            d_left = est_plans[-1] + d_tuple_left[0]
            left = d_tuple_left[1]

            print("d_left: {} d_right: {}".format(d_left, d_right))

            if d_right < d_left:
                print("Направо!")
                mat = right
                est_plans.append(d_right)
                plans.append((-edge[0], -edge[1]))
            else:
                print("Налево!")
                mat = left
                sm.set_index(left_ind)
                plans.append(edge)
                est_plans.append(d_left)
                root.append(edge)

                bool_mat = left.copy().reshape(-1)
                numbers = list(filter(lambda x: x != 0 and x != float('inf'), bool_mat))
                build_root = True if numbers else False

    find = np.where(mat == 0)  # Найти все нулевые элементы
    v_null = zip(find[0], find[1])  # Вектор, содержащий координаты нулевых элементов

    ind = sm.get_index()

    for point in v_null:
        root.append((ind[0][point[0]], ind[1][point[1]]))

    root = sort_root(root)
    final = igs.get_root_estimation(root)
    print(root, final)

    X = [k[0] for k in points]
    Y = [k[1] for k in points]

    X1 = [X[root[i][0]] for i in np.arange(0, len(root), 1)]
    Y1 = [Y[root[i][0]] for i in np.arange(0, len(root), 1)]

    X2 = [X[root[len(root) - 1][0]], X[root[0][0]]]
    Y2 = [Y[root[len(root) - 1][0]], Y[root[0][0]]]

    plt.title('Всего городов: {}\n Координаты X,Y заданы'.format(len(root)))
    plt.plot(X1, Y1, color='r', linestyle=' ', marker='o')
    plt.plot(X1, Y1, color='b', linewidth=1)

    plt.plot(X2[1], Y2[1], color='y', linestyle=' ', marker='o')
    plt.plot(X2, Y2, color='g', linewidth=2, linestyle='-', label='Путь от  последнего \n к первому городу')
    plt.legend(loc='best')

    plt.grid(True)
    plt.show()


