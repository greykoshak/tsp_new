import matplotlib.pyplot as plt
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
# POINTS = [(1, 1), (2, 2)]

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

    def get_split_coordinates(self):
        x = [x[0] for x in self.coordinate_list.values]
        y = [y[1] for y in self.coordinate_list.values]
        return x, y


class GraphScore:
    """ Методы поиска в алгоритме ветвей и границ """

    def __init__(self, df):
        self.df = df
        self.f0 = {"path0": [], "d_min": 0}
        self.f0_estimate()

    def f0_estimate(self):
        """ Первичная оценка нулевого варианта F0=mat(0,1)+mat(1,2)+mat(2,3)+mat(3,4)+mat(4,0)=10+10+20+15+10=65 """

        path = [i for i in range(self.df.shape[0])]
        path.append(path[0])  # Добавить первый элемент в конец списка, чтобы замкнуть маршрут

        for i in range(self.df.shape[0]):
            self.f0["path0"].append((path[i] + 1, path[i + 1] + 1))
            self.f0["d_min"] += self.df.iloc[path[i]][path[i + 1]]

    def get_estimation(self):
        return self.f0

    def get_route_estimation(self, my_route):
        """ Оценка заданного варианта """

        est_path = sum([self.df.loc[x[0]][x[1]] for x in my_route])
        return est_path


class ShowRoute:
    def __init__(self, ref_to_df, show_route):
        self.coord = ref_to_df.get_split_coordinates()
        self.route = show_route
        self.route_coord_x = [self.coord[0][int(self.route[i][0]) - 1] for i in range(len(self.route))]
        self.route_coord_y = [self.coord[1][int(self.route[i][0]) - 1] for i in range(len(self.route))]
        self.route_coord_x2 = [self.coord[0][int(self.route[-1][0]) - 1], self.coord[0][int(self.route[0][0]) - 1]]
        self.route_coord_y2 = [self.coord[1][int(self.route[-1][0]) - 1], self.coord[1][int(self.route[0][0]) - 1]]
        self.show_route()

    def show_route(self):
        plt.title('Всего городов: {}\n Координаты X,Y заданы'.format(len(self.coord[0])))
        plt.plot(self.route_coord_x, self.route_coord_y, color='b', linestyle=' ', marker='o')
        plt.plot(self.route_coord_x, self.route_coord_y, color='b', linewidth=1)

        plt.plot(self.route_coord_x2[1], self.route_coord_y2[1], color='r', linestyle=' ', marker='o')
        plt.plot(self.route_coord_x2, self.route_coord_y2, color='m', linewidth=2, linestyle='--',
                 label='Путь от  последнего \n к первому городу')
        plt.legend(loc='best')

        plt.grid(True)
        plt.show()


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

    zero_pos = df[df.eq(0)].stack().reset_index().values
    inf = float('inf')

    for k in range(zero_pos.shape[0]):
        df.loc[zero_pos[k][0]][zero_pos[k][1]], inf = inf, df.loc[zero_pos[k][0]][zero_pos[k][1]]
        di = df.min(axis=1)[zero_pos[k][0]]
        dj = df.min(axis=0)[zero_pos[k][1]]
        zero_pos[k][2] = di + dj
        df.loc[zero_pos[k][0]][zero_pos[k][1]], inf = inf, df.loc[zero_pos[k][0]][zero_pos[k][1]]
    idx = max(zero_pos, key=lambda el: el[2])
    return idx[0], idx[1]  # Ребро с реальными узлами


def eval_options(eval_df, eval_edge, eval_route):
    # Вариант "вправо" - считаем, что ребро edge не входит в маршрут
    df_right = eval_df.copy()
    df_right.loc[eval_edge[0]][eval_edge[1]] = float('inf')  # Исключаем ребро из маршрута

    eval_right = reduction(df_right)
    d_right, df_right = eval_right[0], eval_right[1]  # Оценка минимума и новая матрица

    # Вариант "влево" - считаем, что ребро edge входит в маршрут
    df_left = eval_df.copy()
    df_left.drop(eval_edge[0], axis=0, inplace=True)  # Delete row
    df_left.drop(eval_edge[1], axis=1, inplace=True)  # Delete column

    # Исключаем ребра, образующие цикл с уже существующим route
    inf_list = forbidden_points(eval_edge, eval_route)

    for p in inf_list:
        if p[0] in df_left.index.values and p[1] in df_left.columns.values:
            df_left.loc[p[0]][p[1]] = float('inf')

    eval_left = reduction(df_left)
    d_left, df_left = eval_left[0], eval_left[1]  # Оценка минимума и новая матрица

    return d_right, df_right, d_left, df_left


def forbidden_points(new_edge: tuple, my_route: list) -> list:
    """ Создать цепочку последовательных звеньев от нового кандидата new_edge """

    path = [new_edge]
    poor_points = list()

    for _ in range(len(my_route)):
        left_item = path[0]
        right_item = path[-1]

        next_item = next(filter(lambda x: right_item[1] == x[0], my_route), False)
        if next_item:
            path.append(next_item)

        prev_item = next(filter(lambda x: left_item[0] == x[1], my_route), False)
        if prev_item:
            path.insert(0, prev_item)

    if path:
        for i in range(len(path)):
            for j in range(len(path) - i):
                poor_points.append((path[j][1], path[i][0]))
    return poor_points


def form_final_route(df_mat_zero, final_route):
    """ Сформировать окончательный маршрут на базе нулевых значений """

    zero_pos = df_mat[df_mat_zero.eq(0)].stack().reset_index().values

    for k in range(zero_pos.shape[0]):
        final_route.append((zero_pos[k][0], zero_pos[k][1]))
    return final_route


def sort_route(unsorted_route: list) -> list:
    """ Создать цепочку последовательных звеньев """

    path = [min(unsorted_route)]
    for _ in range(len(unsorted_route) - 1):
        next_item = path[-1]
        path.append(
            next(filter(lambda x: next_item[1] == x[0], unsorted_route))
        )
    return path


if __name__ == "__main__":
    # class_build_matrix = DataFrameFromMatrix(GIVEN_MATRIX)
    class_build_matrix = DataFrameFromPoints(POINTS)
    df_mat = class_build_matrix.get_df()

    route = list()  # Искомый маршрут комивояжера

    first_pass, build_route = True, True  # build_route: Условие работы цикла: пока есть ненулевые элементы

    while build_route:
        if first_pass:
            graph_score = GraphScore(df_mat)
            f0_dict = graph_score.get_estimation()
            print("\nf0 route is: {}, it's score is: {}".format(f0_dict["path0"], f0_dict["d_min"]))

            d_min_matrix = reduction(df_mat)
            d_min, df_mat = d_min_matrix[0], d_min_matrix[1]  # Оценка минимума минимумов =58 и новая матрица

            if d_min == f0_dict["d_min"]:
                route = f0_dict["path0"]
                break
            first_pass = False
        else:
            edge = graph_edge(df_mat)  # Поиск ребра-кандидата графа
            print("Кандидат: {}".format(edge))
            eval_data = eval_options(df_mat, edge,
                                     route)  # Не забыть записать сумму d_right+d_parent, d_left+d_parent в вектор
            if eval_data[0] < eval_data[2]:
                print("Направо!")
                df_mat = eval_data[1]
                # est_plans.append(d_right)
                # plans.append((-edge[0], -edge[1]))
            else:
                print("Налево!")
                df_mat = eval_data[3]
                # plans.append(edge)
                # est_plans.append(d_left)
                route.append(edge)

                nonzero_arr = df_mat[df_mat.ne(0) & df_mat.ne(float('inf'))].stack().reset_index().values
                build_route = True if nonzero_arr.size != 0 else False

    if d_min < f0_dict["d_min"]:
        route = form_final_route(df_mat, route)
        route = sort_route(route)

        final_est = graph_score.get_route_estimation(route)
        print(route, final_est)

    if isinstance(class_build_matrix, DataFrameFromPoints):
        show = ShowRoute(class_build_matrix, route)
