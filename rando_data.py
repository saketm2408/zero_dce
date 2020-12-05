import numpy as np
import os

if __name__ == "__main__":
    arr = np.ones((1, 100, 100, 3))
    arr = arr.astype(np.float)
    arr.tofile('1.raw')

    with open('list.txt' ,'w') as f:
        for i in range(10):
            f.write('data/1.raw' +  os.linesep)