import torch
import time


"""
Just a few notes for those who know python only through ITSC 1212:

@ performs matrix multiplication

For those who need a refresher on matrix multplication:

assume 

A is:

[1 2 3]
[4 5 6]
[7 8 9]

and B is:

[10 11 12]
[13 14 15]
[16 17 18]

if C is A @ B:

C is:

[1*10 + 2*13 + 3*16   1*11 + 2*14 + 3*17   1*12 + 2*15 + 3*18]
[4*10 + 5*13 + 6*16   4*11 + 5*14 + 6*17   4*12 + 5*15 + 6*18]
[7*10 + 8*13 + 9*16   7*11 + 8*14 + 9*17   7*12 + 8*15 + 9*18]



A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

input_outer_dimension = len(A)
inner_dimension = len(A[0])
outer_dimension = len(B[0])

assert len(A[0]) == len(B) # <- the inner dimension (the number of columns in A and the number of rows in B) must be the same

for i in range(input_outer_dimension):
    for j in range(inner_dimension):
        for k in range(outer_dimension):
            C[i][k] += A[i][j] * B[j][k]


"""

def matrix_multiplication(A, B):
    input_outer_dimension = len(A)
    inner_dimension = len(A[0])
    outer_dimension = len(B[0])

    C = [[0]*outer_dimension for _ in range(input_outer_dimension)]

    assert len(A[0]) == len(B)

    for i in range(input_outer_dimension):
        for j in range(inner_dimension):
            for k in range(outer_dimension):
                C[i][k] += A[i][j] * B[j][k]

    return C




#A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
#B = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]

A = torch.randn(512, 4).tolist()
B = torch.randn(4, 512).tolist()

start_time = time.time()

C = matrix_multiplication(A, B)

end_time = time.time()

print("Python Speed: ", end_time - start_time)

A = torch.tensor(A)
B = torch.tensor(B)

start_time = time.time()

C = A @ B

end_time = time.time()

print("Pytorch Speed: ", end_time - start_time)




