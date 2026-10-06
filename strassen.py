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
        for _ in range(n - r):
            matrix.append([0] * c)
    if c != n:
        for row in matrix:
            for _ in range(n-c):
                row.append(0)

    return matrix

def block(matrix):

    if (len(matrix) <= 2):
        return matrix

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

    if len(a) > 2:
        return [[block(a), block(b)], [block(c), block(d)]]
    return [[a, b], [c, d]]

def unblock(matrix):

    if type(matrix[0][0][0]) == float or type(matrix[0][0][0]) == int:
        return [matrix[0][0] + matrix[1][0], matrix[0][1] + matrix[1][1]]

    unblocked = []
    for block_row in matrix:
        for element in unblock(block_row):
            unblocked.append(element)
    return unblocked

def matrix_add(matrix_a, matrix_b):
    row_vec = False
    if(type(matrix_a[0]) == float or type(matrix_a[0]) == int):
        fixed_a = [matrix_a]
        row_vec = True
    else:
        fixed_a = matrix_a

    new_matrix = []
    for i in range(len(fixed_a)):
        row = []
        for j in range(len(fixed_a[0])):
            row.append(fixed_a[i][j] + matrix_b[i][j])
        new_matrix.append(row)

    if row_vec:
        return new_matrix[0]
    return new_matrix

def matrix_scalar_mult(matrix, s):
    row_vec = False
    if(type(matrix[0]) == float or type(matrix[0]) == int):
        fixed = [matrix]
        row_vec = True
    else:
        fixed = matrix

    copy = []

    for i in range(len(fixed)):
        row = []
        for j in range(len(fixed[0])):
            row.append(s * fixed[i][j])
        copy.append(row)

    if row_vec:
        return copy[0]
    return copy

def recur_mult(matrix_a, matrix_b):

    # prep the matrices by turning them 
    a, b = pad(matrix_a, matrix_b)
    matrix_a = block(a) #TODO: returns weird - eventually matrix_b becomes not 2x2 composed
    matrix_b = block(b)

    # possible that it is concatinating in add and not acutally adding

    scalar = False

    # decide if its a matrix of scalars or of block matrices
    if (type(matrix_a[0][0]) == float or type(matrix_a[0][0]) == int):
        # does it give the same effect
        scalar = True

        # might be a better way to do this with indexing
        a_auxP = [
            (matrix_a[0][0] + matrix_a[1][1]),
            (matrix_a[1][0] + matrix_a[1][1]),
            matrix_a[0][0],
            matrix_a[1][1],
            (matrix_a[0][0] + matrix_a[0][1]),
            (matrix_a[1][0] - matrix_a[0][0]),
            (matrix_a[0][1] - matrix_a[1][1])
        ]

        b_auxP = [
            (matrix_b[0][0] + matrix_b[1][1]),
            matrix_b[0][0],
            (matrix_b[0][1] - matrix_b[1][1]),
            (matrix_b[1][0] - matrix_b[0][0]),
            matrix_b[1][1],
            (matrix_b[0][0] + matrix_b[0][1]),
            (matrix_b[1][0] + matrix_b[1][1])
        ]
    else:
        a_auxP = [ #TODO: look at if these are still working
            matrix_add(matrix_a[0][0], matrix_a[1][1]),
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
    for m in range(7):
        if scalar:
            aux_prod.append(a_auxP[m] * b_auxP[m])
        else:
            aux_prod.append(recur_mult(a_auxP[m], b_auxP[m])) # what iteration is this becoming non-blocked
            # is it going into recur_mult and coming back up for the flag (scoping)?


    # combine the auxilary products to form result matrix entries
    if scalar:
        result = [
            [(aux_prod[0] + aux_prod[3] - aux_prod[4] + aux_prod[6]), (aux_prod[2] + aux_prod[4])],
            [(aux_prod[1] + aux_prod[3]), (aux_prod[0] - aux_prod[1] + aux_prod[2] + aux_prod[5])]
        ]
    else:
        result = [
            [
                matrix_add(
                    matrix_add(
                        aux_prod[0],
                        aux_prod[3]
                    ),
                    matrix_add(
                        matrix_scalar_mult(aux_prod[4], -1),
                        aux_prod[6]
                    )
                ),
                matrix_add(aux_prod[2], aux_prod[4])
            ],
            [
                matrix_add(aux_prod[1], aux_prod[3]),
                matrix_add(
                    matrix_add(
                        aux_prod[0],
                        matrix_scalar_mult(aux_prod[1], -1)
                    ),
                    matrix_add(
                        aux_prod[2],
                        aux_prod[5]
                    )
                )
            ]
        ]

    return result

def sm_mult(matrix_a, matrix_b):
    r = len(matrix_a)
    c = len(matrix_b[0])

    matrix_c = recur_mult(matrix_a, matrix_b) # blocked + padded result matrix
    matrix_c = unblock(matrix_c)

    trimmed_c = []

    # get rid of padding 0s
    for h in range(r):
        row = []
        for e in range(c):
            row.append(matrix_c[h][e])
        trimmed_c.append(row)

    return trimmed_c

def main():
    a = [
        [1, 2, 3, 0, 0, 0],
        [4, 5, 6, 0, 0, 0]
    ]

    b = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    # a = [
    #     [1, 2, 3],
    #     [4, 5, 6]
    # ]

    # b = [
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ]

    #TODO: need a way to handle row vector "matrices"
    # essentially need a way to handle clean edge cases
    # also should check if dimensions can be mult. (this won't throw the same flag as other because padding)
    c = sm_mult(a, b)
    print(c)
    print('blah blah')

    #TODO: put all this stuff into a matrix math file that can be imported

if __name__ == "__main__":
    main()