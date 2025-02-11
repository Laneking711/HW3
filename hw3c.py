import DoolittleMethod as dm
import Gauss_Seidel as gs
import random


def decompose_cholesky(matrix_aug):
    """
    Perform a Cholesky decomposition on the matrix portion of an augmented matrix [A|b], then solve A x = b.

    The function:
      1. Separates the augmented matrix into A and b.
      2. Builds a lower triangular matrix L such that A = L * L^T.
      3. Uses forward substitution (L y = b) and backward substitution (L^T x = y) to find the solution x.

    Args:
        matrix_aug (list of lists): Augmented matrix of the form [A|b].

    Returns:
        tuple: (x, lower, upper)
            x (list): The solution vector to A x = b.
            lower (list): The lower triangular matrix L.
            upper (list): The upper triangular matrix, which is the transpose of L (L^T).
    """

    def compute_transpose(matrix):
        """
        Return the transpose of a 2D list (matrix).

        Args:
            matrix (list of lists): A square or rectangular matrix.

        Returns:
            list of lists: The transpose of the input matrix.
        """
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def forward_substitution(lower, vector):
        """
        Solve lower * y = vector using forward substitution.

        Args:
            lower (list of lists): A lower triangular matrix (n x n).
            vector (list): A right-hand side vector of length n.

        Returns:
            list: The solution vector y of length n.
        """
        n = len(vector)
        y = [0.0] * n
        for i in range(n):
            y[i] = (vector[i] - sum(lower[i][j] * y[j] for j in range(i))) / lower[i][i]
        return y

    def backward_substitution(upper, vector):
        """
        Solve upper * x = vector using backward substitution.

        Args:
            upper (list of lists): An upper triangular matrix (n x n).
            vector (list): A right-hand side vector of length n.

        Returns:
            list: The solution vector x of length n.
        """
        n = len(vector)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (vector[i] - sum(upper[i][j] * x[j] for j in range(i + 1, n))) / upper[i][i]
        return x

    matrix, vector = gs.separateAugmented(matrix_aug)

    # Flatten vector if it is nested
    if isinstance(vector[0], list):
        vector = [item for sublist in vector for item in sublist]

    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]

    # Construct the lower triangular matrix (Cholesky factor)
    for i in range(size):
        for j in range(i + 1):
            if i == j:
                sum_sq = sum(lower[i][k] ** 2 for k in range(j))
                lower[i][j] = (matrix[i][j] - sum_sq) ** 0.5
            else:
                sum_prod = sum(lower[i][k] * lower[j][k] for k in range(j))
                lower[i][j] = (matrix[i][j] - sum_prod) / lower[j][j]

    # Transpose of the lower matrix
    upper = compute_transpose(lower)

    # Solve for x using forward and backward substitution
    y = forward_substitution(lower, vector)
    x = backward_substitution(upper, y)

    return x, lower, upper


def check_symmetric_positive_definite(matrix):
    """
    Check if a matrix is symmetric and positive definite.

    The function first verifies symmetry by comparing matrix[i][j] with matrix[j][i].
    It then generates a random vector and calculates x^T * matrix * x to see if it is strictly positive.

    Args:
        matrix (list of lists): The square matrix to check.

    Returns:
        bool: True if the matrix is symmetric and positive definite, False otherwise.
    """

    def compute_transpose(matrix):
        """
        Return the transpose of a 2D list (matrix).

        Args:
            matrix (list of lists): A square or rectangular matrix.

        Returns:
            list of lists: The transpose of the input matrix.
        """
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

    transpose_matrix = compute_transpose(matrix)
    # Check symmetry
    if matrix != transpose_matrix:
        return False

    # Check positive definiteness
    random_vector = [random.uniform(-1, 1) for _ in range(len(matrix))]
    quadratic_form = sum(
        random_vector[i] * sum(matrix[i][j] * random_vector[j] for j in range(len(matrix)))
        for i in range(len(matrix))
    )

    return quadratic_form > 0


def execute_main():
    """
    Demonstrate solving multiple augmented matrices with either Cholesky or Doolittle.

    1. Predefined augmented matrices are each split into (A, b).
    2. For each matrix, if it is symmetric positive definite, solve by Cholesky.
       Otherwise, solve by Doolittle.
    3. Print the solutions and the corresponding method used.
    """
    matrices = [
        [[1, -1, 3, 2, 15], [-1, 5, -5, -2, -35], [3, -5, 19, 3, 94], [2, -2, 3, 21, 1]],
        [[4, 2, 4, 0, 20], [2, 2, 3, 2, 36], [4, 3, 6, 3, 60], [0, 2, 3, 9, 122]]
    ]

    for index, matrix in enumerate(matrices, start=1):
        A, b = gs.separateAugmented(matrix)

        # Flatten b if it's nested
        if isinstance(b[0], list):
            b = [item for sublist in b for item in sublist]

        if check_symmetric_positive_definite(A):
            solution, _, _ = decompose_cholesky(matrix)
            method_name = "Cholesky"
        else:
            solution = dm.Doolittle(matrix)
            method_name = "Doolittle"

        print(f"Solution for Matrix {index} using {method_name} method: {solution}\n")


if __name__ == "__main__":
    execute_main()
