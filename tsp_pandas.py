import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import pdist, squareform

import tsp_heap

GIVEN_MATRIX = [[0., 10., 25., 25., 10.],
                [1., 0., 10., 15., 2.],
                [8., 9., 0., 20., 10.],
                [14., 10., 24., 0., 15.],
                [10., 8., 25., 27., 0.]]

# Координаты городов
POINTS = [(800, 400), (500, 300), (900, 500), (900, 400), (700, 100), (500, 500), (900, 300), (100, 300)]


class TSP:
    """ Travel Salesman Problem: branch and bound """

    # data_type: 0 - список точек (x, y)
    #          > 0 - матрица расстояний
    def __init__(self, data=None, data_type=0):
        if data is None or not isinstance(data, list):
            raise TypeError("TSP: Do not define correct input data: list of points (x, y) or predefine matrix")
        self.data = data
        self.class_build_matrix = DataFrameFromPoints(data) if data_type == 0 else DataFrameFromMatrix(data)
        self._df_mat = self.class_build_matrix.get_df()

    def run(self):
        """ Find out final route """

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # create the logging file handler
        fh = logging.FileHandler("tsp.log", "w")
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', "%H:%M:%S")
        fh.setFormatter(formatter)

        # add handler to logger object
        logger.addHandler(fh)
        logger.info("Program started {}".format(self.data))

        path_rating = tsp_heap.HeapifyTSP()  # Create empty heap
        route = list()  # Искомый маршрут комивояжера
        first_pass, build_route = True, True  # build_route: Условие работы цикла: пока есть ненулевые элементы

        while build_route:
            if first_pass:
                graph_score = GraphScore(self._df_mat)
                f0_dict = graph_score.get_estimation()  # Route and estimation of f0-approximation
                path_rating.add_element((f0_dict["d_f0"], None))
                print("\nf0 route is: {}, it's score is: {}".format(f0_dict["path0"], f0_dict["d_f0"]))

                d_min_matrix = UtilityTSP.reduction(self._df_mat)
                d_min, self._df_mat = d_min_matrix[0], d_min_matrix[1]  # Оценка минимума минимумов =58 и новая матрица
                d_parent = d_min  # Накопительная оценка предшественников

                if d_min == f0_dict["d_f0"]:
                    route = f0_dict["path0"]
                    break
                first_pass = False
            else:
                edge = UtilityTSP.graph_edge(self._df_mat)  # Поиск ребра-кандидата графа
                print("Кандидат: {}".format(edge))
                eval_data = UtilityTSP.eval_options(self._df_mat, edge, route)
                key_right = d_parent + eval_data[0]
                key_left = d_parent + eval_data[2]
                fl_heap = False

                # Три точки больше f0: f0 - найденный вариант
                if all(map(lambda x: x >= f0_dict["d_f0"],
                           [path_rating.heap[0][0], key_right, key_left])):
                    build_route = False
                # Вершина кучи - следующая точка рассмотрения
                elif all(map(lambda x: path_rating.heap[0][0] < x, [key_right, key_left])):
                    fl_incl = key_left <= key_right
                    path_rating.add_element((key_left if fl_incl else key_right,
                                             SaveState(edge, route, self._df_mat, fl_incl)))

                    edge = path_rating.heap[0][1].edge
                    route = path_rating.heap[0][1].route
                    self._df_mat = path_rating.heap[0][1].matrix
                    fl_incl = path_rating.heap[0][1].include

                    path_rating.del_element()
                    fl_heap = True

                # edge - не включать в маршрут
                if (fl_heap and not fl_incl) or (not fl_heap and key_right < key_left):
                    print("Направо!")
                    # if key_left < f0_dict["d_f0"]:
                    #     path_rating.add_element((key_left,
                    #                             Challenger(edge, route, eval_data[3], True)))
                    self._df_mat = eval_data[1]
                    d_parent = key_right
                # edge - включить в маршрут
                else:
                    print("Налево!")
                    # if key_right < f0_dict["d_f0"]:
                    #     path_rating.add_element((key_right,
                    #                              Challenger(edge, route, eval_data[1], False)))
                    self._df_mat = eval_data[3]
                    d_parent = key_left
                    route.append(edge)

                    nonzero_arr = self._df_mat[
                        self._df_mat.ne(0) & self._df_mat.ne(float('inf'))].stack().reset_index().values
                    build_route = True if nonzero_arr.size != 0 else False

        if d_min < f0_dict["d_f0"]:
            route = UtilityTSP.form_final_route(self._df_mat, route)
            route = UtilityTSP.sort_route(route)

            final_est = graph_score.get_route_estimation(route)
            print(route, final_est)
            logger.info("Final route is \n{}, {}".format(route, final_est))
        else:
            route = f0_dict["path0"]
            final_est = graph_score.get_route_estimation(route)
            print(route, final_est)
            logger.info("Final route is \n{}, {}".format(route, final_est))

        if isinstance(self.class_build_matrix, DataFrameFromPoints):
            ShowRoute(self.class_build_matrix, route)


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
        self.f0 = {"path0": [], "d_f0": 0}
        self.f0_estimate()

    def f0_estimate(self):
        """ Первичная оценка нулевого варианта F0=mat(0,1)+mat(1,2)+mat(2,3)+mat(3,4)+mat(4,0)=10+10+20+15+10=65 """

        path = [i for i in range(self.df.shape[0])]
        path.append(path[0])  # Добавить первый элемент в конец списка, чтобы замкнуть маршрут

        for i in range(self.df.shape[0]):
            self.f0["path0"].append((path[i] + 1, path[i + 1] + 1))
            self.f0["d_f0"] += self.df.iloc[path[i]][path[i + 1]]

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


class UtilityTSP:
    """ Additional methods for a matrix treatment """

    @staticmethod
    def reduction(df):
        """ Редукция строк и столбцов, получение di+dj и новой матрицы после редукций """

        # Get a series containing minimum value of each row
        di = df.min(axis=1)
        df = df.sub(di, axis=0)

        # Get a series containing minimum value of each column
        dj = df.min()
        df = df.sub(dj, axis=1)

        return di.sum() + dj.sum(), df

    @staticmethod
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

    @staticmethod
    def eval_options(eval_df, eval_edge, eval_route):
        # Вариант "вправо" - считаем, что ребро edge не входит в маршрут
        df_right = eval_df.copy()
        df_right.loc[eval_edge[0]][eval_edge[1]] = float('inf')  # Исключаем ребро из маршрута

        eval_right = UtilityTSP.reduction(df_right)
        d_right, df_right = eval_right[0], eval_right[1]  # Оценка минимума и новая матрица

        # Вариант "влево" - считаем, что ребро edge входит в маршрут
        df_left = eval_df.copy()
        df_left.drop(eval_edge[0], axis=0, inplace=True)  # Delete row
        df_left.drop(eval_edge[1], axis=1, inplace=True)  # Delete column

        # Исключаем ребра, образующие цикл с уже существующим route
        inf_list = UtilityTSP.forbidden_points(eval_edge, eval_route)

        for p in inf_list:
            if p[0] in df_left.index.values and p[1] in df_left.columns.values:
                df_left.loc[p[0]][p[1]] = float('inf')

        eval_left = UtilityTSP.reduction(df_left)
        d_left, df_left = eval_left[0], eval_left[1]  # Оценка минимума и новая матрица

        return d_right, df_right, d_left, df_left

    @staticmethod
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

    @staticmethod
    def form_final_route(df_mat_zero, final_route):
        """ Сформировать окончательный маршрут на базе нулевых значений """

        zero_pos = df_mat_zero[df_mat_zero.eq(0)].stack().reset_index().values

        for k in range(zero_pos.shape[0]):
            final_route.append((zero_pos[k][0], zero_pos[k][1]))
        return final_route

    @staticmethod
    def sort_route(unsorted_route: list) -> list:
        """ Создать цепочку последовательных звеньев """

        path = [min(unsorted_route)]
        for _ in range(len(unsorted_route) - 1):
            next_item = path[-1]
            path.append(
                next(filter(lambda x: next_item[1] == x[0], unsorted_route))
            )
        return path


class SaveState:
    """ Remember intermediate state of candidate """

    def __init__(self, v_edge: tuple, v_route: list, v_matrix, v_include: bool):
        self.edge = v_edge
        self.route = v_route
        self.matrix = v_matrix
        self.include = v_include


if __name__ == "__main__":
    aaa = TSP(POINTS, 0)
    # aaa = TSP(GIVEN_MATRIX, 1)
    print("TSP running....")
    aaa.run()
