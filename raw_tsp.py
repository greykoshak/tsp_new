# Travelling salesman problem
import numpy as np
from numpy import sqrt

# from numpy import sqrt

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

        # @staticmethod
        # def matrix_print(matrix):
        #     # Красивый вывод матрицы
        #     for i in range(len(matrix)):
        #         for j in range(len(matrix[i])):
        #             print("{:3.0f}".format(matrix[i][j]), sep=' ', end=' ')
        #         print('', end='\n')


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
        self.set_diagonal()

    def set_diagonal(self):
        """ Диагональные элементы исключаем """
        for i in range(len(self.mat)):
            self.mat[i][i] = float('inf')
        return self.mat


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


def graph_edge(matrix):
    """ Оценка нулевых элементов для поиска ребра графа -кандидата на включение в маршрут """

    find = np.where(matrix == 0)  # Найти все нулевые элементы
    v_null = zip(find[0], find[1])  # Вектор, содержащий координаты нулевых элементов
    max_value = list()  # Оценки нулевых точек
    point = list()  # Координаты нулевых точек
    inf = float('inf')

    for coord in v_null:
        matrix[coord[0], coord[1]], inf = inf, matrix[coord[0], coord[1]]
        di = mat[coord[0]:coord[0] + 1, ].min()  # min по строке
        dj = mat[:, coord[1]].min()  # min по столбцу
        max_value.append(di + dj)
        point.append(coord)
        matrix[coord[0], coord[1]], inf = inf, matrix[coord[0], coord[1]]
    idx = max_value.index(max(max_value))

    return point[idx]


if __name__ == "__main__":
    # mat = DefineMatrix(points).build_matrix()
    # DefineMatrix.matrix_print(mat)

    mat = SetMatrix().set_diagonal()
    # DefineMatrix.matrix_print(mat)

    plans = list()  # Планы
    est_plans = list()  # Оценка планов
    root = list()  # Маршрут комивояжера

    # # Считаем первичную оценку нулевого варианта F0 = mat(0,1) + mat(1,2) + mat(2,3) +
    # # mat(3,4) + mat(4,0) = 10 + 10 +20 + 15 + 10 = 65
    #
    # f0 = 0.0  # Первичная оценка нулевого варианта
    # for i in range(len(mat[0]) - 1):
    #     f0 += mat[i][i + 1]
    #
    # n = len(mat[0])  # Количество точек обхода
    #
    # di = np.zeros((n, 1))  # Одномерный vector для выбора минимального значения по строке
    # dj = np.zeros((1, n))  # Одномерный vector для выбора минимального значения по столбцу
    #
    # di = np.min(mat, axis=1)  # min элемент по строкам
    # di.shape = (n, 1)  # Преобразование вектора-строки в вектор-столбец
    # mat = mat - di  # Редукция строк
    #
    # dj = np.min(mat, axis=0)  # min элемент по столбцам
    # mat = mat - dj  # Редукция столбцов
    #
    # d = di.sum() + dj.sum()
    # est_plans.append(d)

    # Оценка нулевых клеток, поиск ребра для оценки
    # edge = graph_edge(mat)

    igs = InitialGraphScore(mat)
    first_pass = True

    for i in np.arange(0, mat.shape[0], 1):
        if first_pass:
            d_tuple = count_d(mat)
            d, mat = d_tuple[0], d_tuple[1]     # Оценка минимума минимумов =58 и новая матрица
            edge = graph_edge(mat)  # Поиск ребра-кандидата графа
            first_pass = False
        else:
            print(i)
