import numpy as np

DTYPE=np.int8

class Brick(object):
    
    def __init__(self, shape, color=None):
        self.shape = shape
        
        self.rotations = [shape]
        r = shape.copy()
        for _ in range(3):
            present = False
            r = np.rot90(r)
            for rc in self.rotations:
                if r.shape == rc.shape and (r == rc).all():
                    present = True
                    break
            if not present:
                self.rotations.append(r)
        
        self.filters = [
            np.flip(np.flip(r, axis=0), axis=1) for r in self.rotations]
        self.color=color

T = Brick(np.array(
        [
            [0, 1, 0],
            [1, 1, 1]
        ],
        dtype=DTYPE
    ),
    color=1
)

N1 = Brick(
    np.array(
        [
            [0, 1, 1],
            [1, 1, 0]
        ],
        dtype=DTYPE
    ),
    color=2
)


N2 = Brick(
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1]
        ],
        dtype=DTYPE
    ),
    color=3
)

L1 = Brick(
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1]
        ],
        dtype=DTYPE
    ),
    color=4
)


L2 = Brick(
    np.array(
        [
            [0, 0, 1],
            [1, 1, 1]
        ],
        dtype=DTYPE
    ),
    color=5
)

I = Brick(
    np.array(
        [
            [1, 1, 1, 1]
        ],
        dtype=DTYPE
    ),
    color=6
)

B = Brick(
    np.array(
        [
            [1, 1],
            [1, 1]
        ],
        dtype=DTYPE
    ),
    color=7
)

BRICKS = [
    T,
    N1,
    N2,
    L1,
    L2,
    I,
    B
]

if __name__ == '__main__':
    from cl_display import print_matrix
    for c in BRICKS:
        print('-'*80)
        #print(c.__name__)
        for r in c.rotations:
            print_matrix(r)
            print('')
        print('-'*80)
