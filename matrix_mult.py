def matrix_mult(matrix_a, matrix_b):
    is_array = False
    matrix_r = [[0 for i in range(len(matrix_a))] for j in range(len(matrix_b[0]))]

    # this if check does not do what should
    if len(matrix_b[0]) == 1: # how to multiply if only one column
        for i in range(len(matrix_a)):
            for k in range(len(matrix_a[0])):
                matrix_r[0][i] += matrix_a[i][k] * matrix_b[k][0]

    if type(matrix_a[0]) != list:
        matrix_a = [matrix_a]
        is_array = True
        for i in range(len(matrix_a)):
            for j in range(len(matrix_b)):
                for k in range(len(matrix_a[0])):
                    matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]

        if is_array:
            matrix_r = matrix_r[0]

    return matrix_r

def main():
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    b = [
        [1],
        [2],
        [3]
    ]

    result = matrix_mult(a, b)
    for row in result:
        print(row)

if __name__ == "__main__":
    main()