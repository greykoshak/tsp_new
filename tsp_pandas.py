import pandas as pd
import numpy as np

matrix = [(22, 16, 23),
          (33, float('inf'), 11),
          (44, 34, 11),
          (55, 35, float('inf')),
          (66, 36, 13)
          ]

mat = np.array([[float('inf'), 10., 25., 25., 10.],
                [1., float('inf'), 10., 15., 2.],
                [8., 9., float('inf'), 20., 10.],
                [14., 10., 24., float('inf'), 15.],
                [10., 8., 25., 27., float('inf')]])


# Create a DataFrame object
df = pd.DataFrame(mat, index=list('12345'), columns=list('12345'))

print(df)

# # Get a series containing minimum value of each column
dj = df.min()
#
# print('minimum value in each column : ')
print(dj)
#
# # Get a series containing minimum value of each row
di = df.min(axis=1)
#
# print('minimum value in each row : ')
# print(di)

# aa = df.sub(di, axis=0)
# print(aa)

bb = df.sub(dj, axis=1)
print(bb)

sum_vector = di.sum() + dj.sum()
print(sum_vector)

df.drop('2', axis=1, inplace=True)
df.drop('4', axis=0, inplace=True)
print(df)