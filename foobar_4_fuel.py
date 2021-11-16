# Question at bottom of page

def solution(m):
    """
    This function takes an array of arrays of non-negative integers representing how many times each state has jumped to
    each other state and returns an array of integers for each terminal state giving the exact probabilities of each
    terminal state represented as the numerator for each state, then the denominator for all of them at the end and
    in simplest form.
    :param m: array of arrays of non-negative integers
    :return: array of integers
    """

    import fractions
    import math

    # This function takes an array of arrays and returns an array representing the sum of each row
    def get_row_totals(matrix):
        row_totals = []
        for row in matrix:
            row_totals.append(sum(row))
        return row_totals

    # This function turns an array of arrays of integers into an array of arrays of fractions with each row equaling 1
    def get_fraction_matrix(matrix, totals, size):
        fraction_matrix = []
        for i in range(size):
            row = matrix[i]
            fraction_row = []
            for j in range(size):
                numerator = row[j]
                denominator = totals[i]
                if denominator == 0:
                    fraction = 0
                else:
                    fraction = fractions.Fraction(numerator, denominator)
                fraction_row.append(fraction)
            fraction_matrix.append(fraction_row)
        return fraction_matrix

    # This function returns the part of a matrix that references itself
    def get_loop_matrix(matrix, is_not_terminal_list, size):
        loop_matrix = []
        for i in range(size):
            loop_row = []
            for j in range(size):
                if is_not_terminal_list[i] and is_not_terminal_list[j]:
                    loop_row.append(matrix[i][j])
            if is_not_terminal_list[i]:
                loop_matrix.append(loop_row)
        return loop_matrix

    # This function returns the part of a matrix that references terminal states
    def get_terminal_matrix(matrix, is_not_terminal_list, size):
        terminal_matrix = []
        for i in range(size):
            terminal_row = []
            for j in range(size):
                if is_not_terminal_list[i] and not is_not_terminal_list[j]:
                    terminal_row.append(matrix[i][j])
            if is_not_terminal_list[i]:
                terminal_matrix.append(terminal_row)
        return terminal_matrix

    # This function subtracts a matrix from the identity matrix
    def get_inverse_fundamental_matrix(loop_matrix):
        length = len(loop_matrix)
        inv_fund_matrix = []
        for i in range(length):
            inv_fund_row = []
            for j in range(length):
                loop_cell = loop_matrix[i][j]
                if i == j:
                    inv_fund_cell = 1 - loop_cell
                else:
                    inv_fund_cell = 0 - loop_cell
                inv_fund_row.append(inv_fund_cell)
            inv_fund_matrix.append(inv_fund_row)
        return inv_fund_matrix

    def get_transposed_matrix(matrix):
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    # This function takes a matrix and returns another matrix that excludes the row and column specified
    def get_matrix_minor(matrix, row, col):
        return [row[:col] + row[col + 1:] for row in (matrix[:row] + matrix[row + 1:])]

    def get_matrix_determinant(matrix):
        length = len(matrix)

        if length == 2:
            determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            determinant = 0
            for k in range(length):
                determinant += ((-1)**k) * matrix[0][k] * get_matrix_determinant(get_matrix_minor(matrix, 0, k))

        return determinant

    # The determinant returned by this function is not used in the solution, but is kept for code reusability
    def get_matrix_inverse(matrix):

        determinant = get_matrix_determinant(matrix)
        length = len(matrix)

        # base case for 2x2 matrix:
        if length == 2:
            cofactors = [[matrix[1][1], -1 * matrix[0][1]],
                         [-1 * matrix[1][0], matrix[0][0]]]

        # find matrix of cofactors
        else:
            cofactors = []
            for row in range(length):
                cofactor_row = []
                for col in range(length):
                    minor = get_matrix_minor(matrix, row, col)
                    cofactor_row.append(((-1)**(row+col)) * get_matrix_determinant(minor))
                cofactors.append(cofactor_row)
            cofactors = get_transposed_matrix(cofactors)

        return cofactors, determinant

    # This function multiplies two matrices and returns only the first row of the resulting matrix
    def get_first_row_of_matrix_multiplication(matrix_left, matrix_right):
        first_row = []
        for j in range(len(matrix_right[0])):
            cell_total = 0
            for i in range(len(matrix_left[0])):
                cell = matrix_left[0][i] * matrix_right[i][j]
                cell_total += cell
            first_row.append(cell_total)
        return first_row

    # This function takes an array and returns a proportional array of fractions that has all values adding to 1
    def get_probabilities_array(array):
        total = sum(array)
        probabilities_array = [fractions.Fraction(x, total) for x in array]
        return probabilities_array

    # This function takes an array of fractions and returns an array with only their denominators
    def get_denominator_array(fractions_array):
        return [x.denominator for x in fractions_array]

    def least_common_multiple(array):
        lcm = 1
        for i in array:
            lcm = abs(i * lcm) // math.gcd(i, lcm) # USE FRACTIONS FOR PYTHON 2, NOT MATH !!!!!!!!!!!!!!
        return lcm

    # This function takes an array of values and returns an array of fraction numerators with the denominator at the
    # end in simplest form, in which all fractions add to 1
    def get_simplified_probabilities_array(array):

        probabilities_array = get_probabilities_array(array)
        denominator_array = get_denominator_array(probabilities_array)
        lcm = least_common_multiple(denominator_array)

        simplified_prob_array = [int(x.numerator * (lcm / x.denominator)) for x in probabilities_array]
        simplified_prob_array.append(lcm)

        return simplified_prob_array

    # This function takes a matrix of states that link to each other and returns an array with values proportional to
    # the probability of ending in each terminal state
    def solve_markov_chain(matrix, is_not_terminal_list, size):
        loop_matrix = get_loop_matrix(matrix, is_not_terminal_list, size)
        inverse_fundamental_matrix = get_inverse_fundamental_matrix(loop_matrix)
        fundamental_matrix = get_matrix_inverse(inverse_fundamental_matrix)[0]
        terminal_matrix = get_terminal_matrix(matrix, is_not_terminal_list, size)
        resulting_array = get_first_row_of_matrix_multiplication(fundamental_matrix, terminal_matrix)
        return resulting_array

    # Main function
    def calculate_probabilities(matrix):
        size = len(matrix)
        denominators = get_row_totals(matrix)
        if sum(denominators) == 0:
            return [1, 1]
        is_not_terminal_list = [bool(x) for x in denominators]

        fraction_matrix = get_fraction_matrix(matrix, denominators, size)
        resulting_array = solve_markov_chain(fraction_matrix, is_not_terminal_list, size)
        simplified_probabilities_array = get_simplified_probabilities_array(resulting_array)

        return simplified_probabilities_array

    probability = calculate_probabilities(m)
    return probability



test1 = [[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
test2 = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
test3 = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]]
test4 = [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 3, 2, 0, 0], [0, 0, 0, 0, 0, 0]]
test5 = [[163, 562, 564, 354, 651, 79],
         [0, 0, 0, 6, 0, 0],
         [563, 571, 3, 64, 345, 645],
         [38, 345, 1, 78, 677, 2],
         [0, 0, 0, 0, 0, 0],
         [0, 3, 635, 0, 543, 476]]

print(solution(test1))
print(solution(test2))
print(solution(test4))




"""
Doomsday Fuel
=============

Making fuel for the LAMBCHOP's reactor core is a tricky process because of the exotic matter involved. It starts as raw ore, then during processing, begins randomly changing between forms, eventually reaching a stable form. There may be multiple stable forms that a sample could ultimately reach, not all of which are useful as fuel. 

Commander Lambda has tasked you to help the scientists increase fuel creation efficiency by predicting the end state of a given ore sample. You have carefully studied the different structures that the ore can take and which transitions it undergoes. It appears that, while random, the probability of each structure transforming is fixed. That is, each time the ore is in 1 state, it has the same probabilities of entering the next state (which might be the same state).  You have recorded the observed transitions in a matrix. The others in the lab have hypothesized more exotic forms that the ore can become, but you haven't seen all of them.

Write a function solution(m) that takes an array of array of nonnegative ints representing how many times that state has gone to the next state and return an array of ints for each terminal state giving the exact probabilities of each terminal state, represented as the numerator for each state, then the denominator for all of them at the end and in simplest form. The matrix is at most 10 by 10. It is guaranteed that no matter which state the ore is in, there is a path from that state to a terminal state. That is, the processing will always eventually end in a stable state. The ore starts in state 0. The denominator will fit within a signed 32-bit integer during the calculation, as long as the fraction is simplified regularly. 

For example, consider the matrix m:
[
  [0,1,0,0,0,1],  # s0, the initial state, goes to s1 and s5 with equal probability
  [4,0,0,3,2,0],  # s1 can become s0, s3, or s4, but with different probabilities
  [0,0,0,0,0,0],  # s2 is terminal, and unreachable (never observed in practice)
  [0,0,0,0,0,0],  # s3 is terminal
  [0,0,0,0,0,0],  # s4 is terminal
  [0,0,0,0,0,0],  # s5 is terminal
]
So, we can consider different paths to terminal states, such as:
s0 -> s1 -> s3
s0 -> s1 -> s0 -> s1 -> s0 -> s1 -> s4
s0 -> s1 -> s0 -> s5
Tracing the probabilities of each, we find that
s2 has probability 0
s3 has probability 3/14
s4 has probability 1/7
s5 has probability 9/14
So, putting that together, and making a common denominator, gives an answer in the form of
[s2.numerator, s3.numerator, s4.numerator, s5.numerator, denominator] which is
[0, 3, 2, 9, 14].

Languages
=========

To provide a Java solution, edit Solution.java
To provide a Python solution, edit solution.py

Test cases
==========
Your code should pass the following test cases.
Note that it may also be run against hidden test cases not shown here.

-- Java cases --
Input:
Solution.solution({{0, 2, 1, 0, 0}, {0, 0, 0, 3, 4}, {0, 0, 0, 0, 0}, {0, 0, 0, 0,0}, {0, 0, 0, 0, 0}})
Output:
    [7, 6, 8, 21]

Input:
Solution.solution({{0, 1, 0, 0, 0, 1}, {4, 0, 0, 3, 2, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}})
Output:
    [0, 3, 2, 9, 14]

-- Python cases --
Input:
solution.solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]])
Output:
    [7, 6, 8, 21]

Input:
solution.solution([[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
Output:
    [0, 3, 2, 9, 14]

Use verify [file] to test your solution and see how it does. When you are finished editing your code, use submit [file] to submit your answer. If your solution passes the test cases, it will be removed from your home folder.
"""