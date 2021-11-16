import numpy as np


if __name__ == '__main__':
    X1 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    X2 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])

    print('X1==5 ===================')
    print(X1==5)
    print('X2==1 ===================')
    print(X2==1)
    logicaled_X = np.logical_and(X1>=5, X2==1)
    print('logicaled_X ===================')
    print(logicaled_X)

    segmap = np.zeros(shape=(3, 3), dtype=np.uint8)
    print(segmap)
    segmap[0], segmap[1] = 255, 255
    print(segmap)
    segmap[logicaled_X] = 2
    print(segmap)