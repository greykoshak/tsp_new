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

    @staticmethod
    def matrix_print(matrix):
        # Красивый вывод матрицы
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                print("{:3.0f}".format(matrix[i][j]), sep=' ', end=' ')
            print('', end='\n')


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
        for i in range(len(self.mat)):
            self.mat[i][i] = float('inf')
        return self.mat


if __name__ == "__main__":
    # mat = DefineMatrix(points).build_matrix()
    # DefineMatrix.matrix_print(mat)

    mat = SetMatrix().set_diagonal()
    # DefineMatrix.matrix_print(mat)

    # Считаем первичную оценку нулевого варианта F0 = mat(0,1) + mat(1,2) + mat(2,3) +
    # mat(3,4) + mat(4,0) = 10 + 10 +20 + 15 + 10 = 65

    f0 = 0.0  # Первичная оценка нулевого варианта
    for i in range(len(mat[0]) - 1):
        f0 += mat[i][i + 1]
    f0 += mat[len(mat[0]) - 1][0]

    n = len(mat[0])  # Количество точек обхода

    di = np.zeros((n, 1))  # Одномерный vector для выбора минимального значения по строке
    dj = np.zeros((1, n))  # Одномерный vector для выбора минимального значения по столбцу

    # for i in np.arange(0, n, 1):
    #     di[i][0] = mat[i:i + 1, ].min()
    #     dj[0][i] = mat[:, i].min()

    di = np.min(mat, axis=1)  # min элемент по строкам
    di.shape = (n, 1)   # Преобразование вектора-строки в вектор-столбец
    mat = mat - di  # Вычесть из каждого столбца матрицы mat вектор di

    dj = np.min(mat, axis=0)  # min элемент по столбцам
    mat = mat - dj  # Вычесть из каждой строки матрицы mat вектор dj
    print(mat)
