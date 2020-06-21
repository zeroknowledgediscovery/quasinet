import sys
import copy

import numpy as np
from numba import int32, float32    # import the types
from numba.experimental import jitclass
from numba import njit
from numba.core import types
from numba.typed import Dict, List



sys.path.insert(1, '/home/jinli11/quasinet/quasinet/citrees/')

from utils import remove_zeros

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

# @jitclass(spec)
# class Bag(object):
#     def __init__(self, value):
#         self.value = value
#         self.array = np.zeros(value, dtype=np.float32)

#     @property
#     def size(self):
#         return self.array.size

#     def increment(self, val):
#         for i in range(self.size):
#             self.array[i] += val
#         return self.array

#     @staticmethod
#     def add(x, y):
#         return x + y

# n = 21
# mybag = Bag(n)

# @njit(cache=True, nogil=True, fastmath=True)
# def create_chi2_table(x, y):
#     """Create a chi-squared contingency table using x and y
#     """

#     chi2_table = np.zeros( ( np.max(y) + 1, np.max(x) + 1), dtype=np.int32)

#     for i in np.arange(x.shape[0]):
#         chi2_table[y[i], x[i]] += 1

#     return chi2_table

# x = np.random.randint(0, 5, 100)#.reshape(-1, 1)
# y = np.random.randint(0, 6, 100)#.reshape(-1, 1)
# table = create_chi2_table(x, y)



# breakpoint()

# The Dict.empty() constructs a typed dictionary.
# The key and value typed must be explicitly declared.
d = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

# The typed-dict can be used from the interpreter.
d['posx'] = np.asarray([1, 0.5, 2], dtype='f8')
d['posy'] = np.asarray([1.5, 3.5, 2], dtype='f8')
d['velx'] = np.asarray([0.5, 0, 0.7], dtype='f8')
d['vely'] = np.asarray([0.2, -0.2, 0.1], dtype='f8')

d2 = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

# The typed-dict can be used from the interpreter.
d2['posx'] = np.asarray([1, 0.5, 2], dtype='f8')
d2['posy'] = np.asarray([1.5, 3.5, 2], dtype='f8')
d2['velx'] = np.asarray([0.5, 0, 0.7], dtype='f8')
d2['vely'] = np.asarray([0.2, -0.2, 0.1], dtype='f8')

# Here's a function that expects a typed-dict as the argument
@njit
def move(ds):
    # inplace operations on the arrays
    # d['posx'] += d['velx']
    # d['posy'] += d['vely']
    
    total = 0
    for d  in ds:
        for k, v in d.items():
            total += v[0]
        # total += len(d)
    return total

print('posx: ', d['posx'])  # Out: posx:  [1.  0.5 2. ]
print('posy: ', d['posy'])  # Out: posy:  [1.5 3.5 2. ]

# Call move(d) to inplace update the arrays in the typed-dict.
# list_d = [d, d2]
l = List()
l.append(d)
l.append(d2)
print (move(l))

print('posx: ', d['posx'])  # Out: posx:  [1.5 0.5 2.7]
print('posy: ', d['posy'])  # Out: posy:  [1.7 3.3 2.1]


# from numba import njit, prange

# @njit(cache=True, nogil=True, fastmath=True)
# def mad_function(x):
#     return (x ** 3.423432) * 2.4

# @njit(parallel=True)
# def prange_test(As):
#     # s = 0.0

#     total_s = 0.0
#     for A in As:
#         s = np.empty((A.shape[0], A.shape[0]))
#         # Without "parallel=True" in the jit-decorator
#         # the prange statement is equivalent to range
#         for i in prange(A.shape[0]):
#             # s[i] = A[i] * 4.2
#             for j in np.arange(A.shape[0]):
#                 s[i, j] = mad_function(A[i])
#         total_s += np.sum(s)

#     return total_s

# z = np.array([2.2, 21.3]).astype(np.float32) #.reshape(-1, 1)
# result = prange_test([z, z])
# print (result)


# @njit(cache=True, parallel=True)
# def ident_parallel(x):
#     return np.cos(x) ** 2 + np.sin(x) ** 2


# import time

# start = time.time()

# z = ident_parallel(np.random.uniform(size=(10000000,)))

# end = time.time()
# print(end - start)

# print (z)