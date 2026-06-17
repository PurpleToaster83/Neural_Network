def matrix_mult(matrix_a, matrix_b):
    is_array = False
    a = len(matrix_a)

    if type(matrix_a[0]) != list:
        is_array = True
        a = 1
        matrix_a = [matrix_a]

    matrix_r = [[0 for i in range(len(matrix_b[0]))] for j in range(a)]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_a[0])):
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]

    if is_array:
        matrix_r = matrix_r[0]

    return matrix_r

def main():
    a = [
        [6, 2],
        [1, 3]
    ]

    b = [
        [1, 2, 3],
        [2, 4, 5]
    ]
    print(len(a))
    print(len(b[0]))

    result = matrix_mult(a, b)
    for row in result:
        print(row)
    print('blah')

if __name__ == "__main__":
    main()