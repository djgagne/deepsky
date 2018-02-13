import numpy as np
from multiprocessing import Pool, Array
import ctypes
import traceback


def main():
    data_shape = (10000, 32, 32, 1)
    global data
    data = np.random.random(size=data_shape)
    pool = Pool(2)
    output = []
    for i in range(0, 100, 10):
        pool.apply_async(child, (i, ), callback=output.append)
    pool.close()
    pool.join()
    for o in output:
        print(o)
    return


def child(r):
    try:
        return data[r].sum()
    except Exception as e:
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()

