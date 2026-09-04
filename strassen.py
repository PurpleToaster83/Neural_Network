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

    for ro, row in enumerate(matrix):

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
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            row.append(matrix_a[i][j] + matrix_b[i][j])
        new_matrix.append(row)
    return new_matrix

def matrix_scalar_mult(matrix, s):
    copy = []

    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(s * matrix[i][j])
        copy.append(row)

    return copy

def sm_mult(matrix_a, matrix_b):

    # prep the matrices
    a, b = pad(matrix_a, matrix_b)
    matrix_a = block(a)
    matrix_b = block(b) #TODO: need to fix this prep stuff to hand the scalar vs block matrix question

    scalar = False

    # decide if its a matrix of scalars or of block matrices
    if len(matrix_a[0][0][0]) == 1:
        scalar = True

        # might be a better way to do this with indexing
        a_auxP = [
            (matrix_a[0][0][0][0] + matrix_a[1][1][0][0]),
            (matrix_a[1][0][0][0] + matrix_a[1][1][0][0]),
            matrix_a[0][0][0][0],
            matrix_a[1][1][0][0],
            (matrix_a[0][0][0][0] + matrix_a[0][1][0][0]),
            (matrix_a[1][0][0][0] - matrix_a[0][0][0][0]),
            (matrix_a[0][1][0][0] - matrix_a[1][1][0][0])
        ]

        b_auxP = [
            (matrix_b[0][0][0][0] + matrix_b[1][1][0][0]),
            matrix_b[0][0][0][0],
            (matrix_b[0][1][0][0] - matrix_b[1][1][0][0]),
            (matrix_b[1][0][0][0] - matrix_b[0][0][0][0]),
            matrix_b[1][1][0][0],
            (matrix_b[0][0][0][0] + matrix_b[0][1][0][0]),
            (matrix_b[1][0][0][0] + matrix_b[1][1][0][0])
        ]
    else: 
        a_auxP = [
            matrix_add(matrix_a[0][0], matrix_a[1][1]), # I don't think matrix add is working
            matrix_add(matrix_a[1][0], matrix_a[1][1]),
            matrix_a[0][0],
            matrix_a[1][1],
            matrix_add(matrix_a[0][0], matrix_a[0][1]),
            matrix_add(matrix_a[1][0], matrix_scalar_mult(matrix_a[0][0], -1)),
            matrix_add(matrix_a[0][1], matrix_scalar_mult(matrix_a[1][1], -1))
        ]

        b_auxP = [
            matrix_add(matrix_b[0][0], matrix_b[1][1]),
            matrix_b[0][0],
            matrix_add(matrix_b[0][1], matrix_scalar_mult(matrix_b[1][1], -1)),
            matrix_add(matrix_b[1][0], matrix_scalar_mult(matrix_b[0][0], -1)),
            matrix_b[1][1],
            matrix_add(matrix_b[0][0], matrix_b[0][1]),
            matrix_add(matrix_b[1][0], matrix_b[1][1])
        ]

    aux_prod = []

    # apply hadamard product operation
    for m in range(8):
        if scalar:
            aux_prod.append(a_auxP[m] * b_auxP[m])
        else:
            aux_prod.append(sm_mult(a_auxP[m], b_auxP[m])) #TODO: this line is where it enters infinite recursion
            # check but, think need to "unwrap" more if not scalar
            # matrix_a and matrix_b do not change between iterations --> cause scalar add not working ;)

    # combine the auxilary products to form result matrix entries
    if scalar:
        result = [
            [(aux_prod[0] + aux_prod[1]), (aux_prod[4] - aux_prod[6])],
            [(aux_prod[2] + aux_prod[5]), (aux_prod[4] + aux_prod[5] - aux_prod[1] - aux_prod[3])]
        ]
    else:
        result = [
            [matrix_add(aux_prod[0], aux_prod[1]), matrix_add(aux_prod[4], matrix_scalar_mult(aux_prod[6], -1))],
            [matrix_add(aux_prod[2], aux_prod[5]), matrix_add(matrix_add(aux_prod[4] + aux_prod[5]), matrix_scalar_mult(matrix_add(aux_prod[1], aux_prod[3]), -1))]
        ]

    return result
    #TODO: crashes (likely unbounded recursion)
    # means that scalar checks not working

def main():
    a = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    b = [
        [1, 2, 3],
        [4, 5, 6]
    ]

    #TODO: need a way to handle row vector "matrices"
    # essentially need a way to handle clean edge cases
    # also should check if dimensions can be mult. (this won't throw the same flag as other because padding)
    c = sm_mult(a, b)

    print('blah')


if __name__ == "__main__":
    main()