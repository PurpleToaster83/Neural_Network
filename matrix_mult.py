def matrix_mult(matrix_a, matrix_b):
    matrix_r = [[0 for i in range(len(matrix_a))] for j in range(len(matrix_b[0]))]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b)):
            for k in range(len(matrix_a[0])):
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return matrix_r

def main():
    a = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    b = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    result = matrix_mult(a, b)
    for row in result:
        print(row)

if __name__ == "__main__":
    main()