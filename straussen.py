import math

def pad(matrix_a, matrix_b): 
    # determine largest dimension amoung the two matrices
    max_d = max(len(matrix_a), len(matrix_a[0]), len(matrix_b), len(matrix_b[0]))

    # the side must be a power of 2
    power = max_d.bit_length() - 1 # look for the MSB of max_d
    if max_d % pow(2, power) != 0:
        max_d = pow(2, power + 1)

    return (square(matrix_a, max_d), square(matrix_b, max_d))

def square(matrix, n):

    # determine dimensions of matrix
    r = len(matrix)
    c = len(matrix[0])

    # add zero padding if not 2 power to rows or columns
    if r != n:
        new_r = ([0] * (c - 1))
        for _ in range(n - r):
            matrix.append(new_r)
    if c != n:
        for _ in range(n - c):
            for row in matrix:
                row.append(0)

    return matrix

def block(matrix): 

    # determine dimensions of matrix
    r = len(matrix)
    col = len(matrix[0])

    a = []
    b = []
    c = []
    d = []

    for ro, row in enumerate(matrix): #TODO: fix this logic

        one_par = []
        two_par = []

        # segregate between left and right
        for co, el in enumerate(row):
            if (co <= (col / 2) - 1):
                one_par.append(el)
            else:
                two_par.append(el)

        # segregate left and right based on up or down
        if (ro <= (r / 2) - 1):
            a.append(one_par)
            b.append(two_par)
        else:
            c.append(one_par)
            d.append(two_par)

    return [[a, b], [c, d]]

def matrix_add(matrix_a, matrix_b):
    new_matrix = []
    for e in range(len(matrix_a)):
        new_matrix.append(matrix_a[e] + matrix_b[e])
    return new_matrix

def matrix_scalar_mult(matrix, s):
    copy = matrix

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            copy[i][j] *= s

    return copy

def sm_mult(matrix_a, matrix_b):

    # prep the matrices
    a, b = pad(matrix_a, matrix_b)
    matrix_a = block(a)
    matrix_b = block(b) #TODO: need to fix this prep stuff to hand the scalar vs block matrix question

    # a_auxP = [
    #     matrix_add(matrix_a[0][0], matrix_a[1][1]),
    #     matrix_add(matrix_a[1][0], matrix_a[1][1]),
    #     matrix_a[0][0],
    #     matrix_a[1][1],
    #     matrix_add(matrix_a[0][0], matrix_a[0][1]),
    #     matrix_add(matrix_a[1][0], matrix_scalar_mult(matrix_a[0][0], -1)),
    #     matrix_add(matrix_a[0][1], matrix_scalar_mult(matrix_a[1][1], -1))
    # ]

    # b_auxP = [
    #     matrix_add(matrix_b[0][0], matrix_b[1][1]),
    #     matrix_b[0][0],
    #     matrix_add(matrix_b[0][1], matrix_scalar_mult(matrix_b[1][1], -1)),
    #     matrix_add(matrix_b[1][0], matrix_scalar_mult(matrix_b[0][0], -1)),
    #     matrix_b[1][1],
    #     matrix_add(matrix_b[0][0], matrix_b[0][1]),
    #     matrix_add(matrix_b[1][0], matrix_b[1][1])
    # ]

    a_inst = [
        [(0,0), (1,1)],
        [(1,0), (1,1)],
        [(0,0)],
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (0,0)], #negative
        [(0,1), (1,1)]  #negative
    ]

    b_inst = [
        [(0,0), (1,1)],
        [(0,0)],
        [(0,1), (1,1)], #negative
        [(1,0), (0,0)], #negative
        [(1,1)],
        [(0,0), (0,1)],
        [(1,0), (1,1)]
    ]

    aux_prod = []

    scalar = type(matrix_a[0][0]) == int
    
    for m in range(8):
        pass

def main():
    a = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    b = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    d = [
        [1, 2],
        [3, 4]
    ]

    #TODO: need a way to handle row vector "matrices"
    # essentially need a way to handle clean edge cases
    c = sm_mult(a, b)

    # need to check type before multiply or add to see if need to do recursion

    print('blah')


if __name__ == "__main__":
    main()