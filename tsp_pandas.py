import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

GIVEN_MATRIX = [[0., 10., 25., 25., 10.],
                [1., 0., 10., 15., 2.],
                [8., 9., 0., 20., 10.],
                [14., 10., 24., 0., 15.],
                [10., 8., 25., 27., 0.]]

# Координаты городов
POINTS = [(800, 400), (500, 300), (900, 500), (900, 400), (700, 100), (500, 500), (900, 300), (100, 300)]


class DataFrameFromMatrix:
    """ Матрица расстояний задана """

    def __init__(self, matrix):
        self.matrix = pd.DataFrame(matrix,
                                   index=list(map(str, [i + 1 for i in range(len(matrix))])),
                                   columns=list(map(str, [i + 1 for i in range(len(matrix))])))

    def get_df(self):
        self.matrix.values[tuple([np.arange(self.matrix.shape[0])] * 2)] = float('inf')
        return self.matrix


class DataFrameFromPoints:
    """ Calculate distance matrix between 2D-points """

    def __init__(self, coordinate_list):
        self.coordinate_list = pd.DataFrame(coordinate_list)
        self.matrix = None

    def get_df(self):
        """ pdist заполняет матрицу сверху диагонали, squareform - заполняет нижнюю часть """

        distances = pdist(self.coordinate_list.values, metric='euclidean')  # Returns: ndarray
        dist_matrix = squareform(distances)
        dist_matrix = np.round(dist_matrix, 2)
        self.matrix = pd.DataFrame(dist_matrix,
                                   index=list(map(str, [i + 1 for i in range(len(dist_matrix))])),
                                   columns=list(map(str, [i + 1 for i in range(len(dist_matrix))])))
        self.matrix.values[tuple([np.arange(self.matrix.shape[0])] * 2)] = float('inf')

        return self.matrix


class GraphScore:
    """ Методы поиска в алгоритме ветвей и границ """

    def __init__(self, df):
        self.df = df
        self.f0 = {"path0": [], "d_min": 0}
        self.f0_estimate()

    def f0_estimate(self):
        """ Первичная оценка нулевого варианта F0=mat(0,1)+mat(1,2)+mat(2,3)+mat(3,4)+mat(4,0)=10+10+20+15+10=65 """

        path = [i for i in range(self.df.shape[0])]
        path.append(path[0])

        for i in range(self.df.shape[0]):
            self.f0["path0"].append((path[i], path[i + 1]))
            self.f0["d_min"] += self.df.iloc[path[i]][path[i + 1]]

    def get_estimation(self):
        return self.f0

    def get_root_estimation(self, my_root):
        """ Оценка заданного варианта """

        for pnt in my_root:
            self.final += self.mat[pnt[0]][pnt[1]]
        return self.final


def reduction(df):
    """ Редукция строк и столбцов, получение di+dj и новой матрицы после редукций """

    # Get a series containing minimum value of each row
    di = df.min(axis=1)
    df = df.sub(di, axis=0)

    # Get a series containing minimum value of each column
    dj = df.min()
    df = df.sub(dj, axis=1)

    return di.sum() + dj.sum(), df


def graph_edge(df):
    """ Оценка нулевых элементов для поиска ребра графа -кандидата на включение в маршрут """

    result = np.where(df == 0)  # Найти все нулевые элементы
    _v_null = zip(result[0], result[1])  # Вектор, содержащий координаты нулевых элементов
    max_value = list()  # Оценки нулевых точек
    pnt = list()  # Координаты нулевых точек
    inf = float('inf')

    for k in _v_null:
        df.iloc[k[0]][k[1]], inf = inf, df.iloc[k[0]][k[1]]
        print(k[0])
        di = df.iloc[k[0]].min(axis=1)
        print(di)
        dj = df.iloc[k[1]].min()
        max_value.append(di + dj)
        pnt.append((k[0], k[1]))
        df.iloc[k[0]][k[1]], inf = inf, df.iloc[k[0]][k[1]]
    print(max_value)
    idx = max_value.index(max(max_value))
    return pnt[idx]  # Ребро с реальными узлами


if __name__ == "__main__":
    class_build_matrix = DataFrameFromMatrix(GIVEN_MATRIX)
    df_mat = class_build_matrix.get_df()

    root = list()  # Искомый маршрут комивояжера

    first_pass, build_root = True, True  # build_root: Условие работы цикла: пока есть ненулевые элементы

    while build_root:
        if first_pass:
            graph_score = GraphScore(df_mat)
            f0_dict = graph_score.get_estimation()
            print("f0 root is: {}, it's score is: {}".format(f0_dict["path0"], f0_dict["d_min"]))

            d_min_matrix = reduction(df_mat)
            d_min, df_mat = d_min_matrix[0], d_min_matrix[1]  # Оценка минимума минимумов =58 и новая матрица
            print("d_min: {}".format(d_min))

            if d_min == f0_dict["d_min"]:
                root.append(f0_dict["path0"])
                break

            first_pass = False
        else:
            edge = graph_edge(df_mat)  # Поиск ребра-кандидата графа
            print(edge)

            build_root = False  # True if there_is_nonzero else False

# mm = DataFrameFromMatrix(mat)
# df = mm.get_df()

# mm = DataFrameFromPoints(points)
# df = mm.get_df()

# print(df)

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
