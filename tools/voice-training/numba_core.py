"""numba port of piper's Cython monotonic alignment (core.pyx), so no C compiler is needed."""

import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def _maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            v_prev = (0.0 if y == 0 else max_neg_val) if x == 0 else value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if y > 0 and index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@numba.njit(nogil=True, cache=True, parallel=True)
def maximum_path_c(paths, values, t_ys, t_xs):
    for i in numba.prange(paths.shape[0]):
        _maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


__all__ = ["maximum_path_c", "np"]
