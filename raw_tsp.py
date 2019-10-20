# Travelling salesman problem
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
        return sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    def get_number_city(self):
        return self.number


class SetMatrix:
    """ Использовать заданную матрицу расстояний """

    def __init__(self):
        self.mat = np.array([[-1., 10., 25., 25., 10.],
                             [1., -1, 10., 15., 2.],
                             [8., 9., -1., 20., 10.],
                             [14., 10., 24., -1., 15.],
                             [10., 8., 25., 27., -1.]])
        self.index = [[j for j in range(len(self.mat))] for i in range(2)]  # Индексы строк и столбцов
        self.set_diagonal()

    def set_diagonal(self):
        """ Диагональные элементы исключаем """
        for _i in range(len(self.mat)):
            self.mat[_i][_i] = float('inf')
        return self.mat

    def get_index(self):
        return self.index

    def set_index(self, update_list):
        self.index = update_list

    def mod_index(self, edge: tuple) -> tuple:
        func = lambda x: (self.index[0].index(x[0]), self.index[1].index(x[1]))
        return func(edge)


class InitialGraphScore:
    """ Методы поиска в алгоритме ветвей и границ """

    def __init__(self, matrix):
        self.mat = matrix
        self.f0 = 0.
        self.f0_root = list()

        self.f0_estimate()

    def f0_estimate(self):
        # Первичная оценка нулевого варианта F0=mat(0,1)+mat(1,2)+mat(2,3)+mat(3,4)+mat(4,0)=10+10+20+15+10=65
        for i in range(len(mat[0]) - 1):
            self.f0 += self.mat[i][i + 1]
            self.f0_root.append((i, i + 1))
        self.f0 += self.mat[len(self.mat[0]) - 1][0]  # f0 = 65
        self.f0_root.append((len(self.mat[0]) - 1, 0))


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


def graph_edge(matrix, ind):
    """ Оценка нулевых элементов для поиска ребра графа -кандидата на включение в маршрут """

    find = np.where(matrix == 0)  # Найти все нулевые элементы
    v_null = zip(find[0], find[1])  # Вектор, содержащий координаты нулевых элементов
    max_value = list()  # Оценки нулевых точек
    point = list()  # Координаты нулевых точек
    inf = float('inf')

    for coords in v_null:
        matrix[coords[0], coords[1]], inf = inf, matrix[coords[0], coords[1]]
        di = mat[coords[0]:coords[0] + 1, ].min()  # min по строке
        dj = mat[:, coords[1]].min()  # min по столбцу
        max_value.append(di + dj)
        point.append((ind[0][coords[0]], ind[1][coords[1]]))
        matrix[coords[0], coords[1]], inf = inf, matrix[coords[0], coords[1]]
    idx = max_value.index(max(max_value))

    return point[idx]  # Ребро с реальными узлами


def select_wrong_root(root: list, edge: tuple) -> list:
    l_root = root[:]

    l_root.append(edge)
    l_root.sort()

    print("root: l_root: ", l_root, "edge: ", edge)
    wrong_root = [(edge[1], edge[0])]
    print("--- wrong roots: {} ---".format(wrong_root))

    return wrong_root


def set_value(matrix, edge, value):
    """ Для заданной ячейки установить столбец и строку значение """

    matrix[edge[0]:edge[0] + 1, ] = value
    matrix[:, edge[1]] = value

    return matrix


if __name__ == "__main__":
    # mat = DefineMatrix(points).build_matrix()
    # DefineMatrix.matrix_print(mat)

    sm = SetMatrix()
    mat = sm.set_diagonal()
    ind = sm.get_index()

    plans = list()  # Планы
    est_plans = list()  # Оценка планов
    root = list()  # Маршрут комивояжера

    igs = InitialGraphScore(mat)
    first_pass = True

    for i in np.arange(0, mat.shape[0], 1):
        print("\n============ i = {} ================\n".format(i))
        if first_pass:
            d_tuple = count_d(mat)
            d_min, mat = d_tuple[0], d_tuple[1]  # Оценка минимума минимумов =58 и новая матрица
            est_plans.append(d_min)

            if d_min == igs.f0:
                root.append(igs.f0_root)
                break
            first_pass = False
        else:
            print("index: {}: ".format(sm.get_index()))
            print(mat)
            edge = graph_edge(mat, ind)  # Поиск ребра-кандидата графа (реальные узлы)
            print("edge = {}".format(edge))

            # Вариант "вправо" - считаем, что ребро edge не входит в маршрут
            right = mat.copy()
            ii = sm.mod_index(edge)
            print("ii = {}".format(ii))
            right[ii] = float('inf')  # Исключаем ребро из маршрута
            d_tuple_right = count_d(right)
            d_right = est_plans[-1] + d_tuple_right[0]

            # Вариант "влево" - считаем, что ребро edge входит в маршрут
            left = mat.copy()
            left = set_value(left, edge, 0.)
            inf_list = select_wrong_root(root, edge)  # Исключаем ребра, образующие цикл с уже существующим root

            for coord in inf_list:
                left[coord] = float('inf')

            d_tuple_left = count_d(left)
            d_left = est_plans[-1] + d_tuple_left[0]

            print("d_left: {} d_right: {} root {}".format(d_left, d_right, root))

            if d_right < d_left:
                mat = right
                est_plans.append(d_right)
                root.append((-ii[0], -ii[1]))
                plans.append((-ii[0], -ii[1]))
            else:
                plans.append(edge)
                est_plans.append(d_left)
                root.append((ii[0], ii[1]))

                left = np.delete(left, (ind[0][edge[0]]), axis=0)
                mat = np.delete(left, (ind[1][edge[1]]), axis=1)

                del ind[0][edge[0]]
                del ind[1][edge[1]]
                sm.set_index(ind)
