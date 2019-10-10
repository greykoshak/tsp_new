# Travelling salesman problem
import numpy as np
from numpy import sqrt

# from numpy import sqrt

points = [(800, 400), (500, 300), (900, 500), (900, 400), (700, 100), (500, 500), (900, 300), (100, 300)]
X = [k[0] for k in points]
Y = [k[1] for k in points]

n = len(X)

for ib in np.arange(0, n, 1):
    M = np.zeros([n, n])


class City():
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


if __name__ == "__main__":

    n = len(points)  # Размерность матрицы
    cities = [City(i, point[0], point[1]) for i, point in enumerate(points)]
    distances = np.zeros((n, n))

    for i in np.arange(0, n, 1):
        for j in np.arange(i, n, 1):
            if i == j:
                distances[i, j] = float('inf')
                continue

            distances[i, j] = City.distance(cities[i], cities[j])
            distances[j][i] = distances[i][j]

    print(distances)