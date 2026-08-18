def add_empty(matrix):

    # determine dimensions of matrix
    r = len(matrix)
    c = len(matrix[0])

    # add zero padding if not even number rows or columns
    if r % 2 != 0:
        new_r = ([0] * c)
        matrix.append(new_r)
    if c % 2 != 0:
        for row in matrix:
            row.append(0)

    return matrix

def block(matrix):

    # determine dimensions of matrix
    r = len(matrix)
    c = len(matrix[0])

    a = []
    b = []
    c = []
    d = []

    for ro, row in enumerate(matrix):

        one_par = []
        two_par = []

        # segregate between left and right
        for co, el in enumerate(row):
            if (co <= c / 2):
                one_par.append(el)
            else:
                two_par.append(el)

        # segregate left and right based on up or down
        if (ro <= r / 2):
            a.append(one_par)
            b.append(two_par)
        else:
            c.append(one_par)
            d.append(two_par)

    return [[a, b], [c, d]]  #TODO: needs to make into square matrix not just even         

def main():
    a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    b = [
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ]

    a = block(add_empty(a))
    b = block(add_empty(b))

    print('blah')


if __name__ == "__main__":
    main()