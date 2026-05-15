import numpy as np

A = np.array([[1, 4, 9],
              [4 ,9 , 16],
              [9, 16, 25]
              ])

b = np.array([[10],
             [14], 
             [18]]
             )

def gauss_elimination_with_pivoting(A, b):
    n = np.size(A, axis = 0)

    augmented_matrix = np.concatenate((A, b), axis = 1)
    
    for j in range(n):
        pivot_row = j
        
        max = abs(augmented_matrix[j, j])
        
        for k in range(j + 1, n):
            if abs(augmented_matrix[k, j]) > max:
                max = abs(augmented_matrix[k, j])
                pivot_row = k
        if max == 0:
            print("Singular Matrix -> Stop Program")
            return None
        
        if pivot_row != j:
            augmented_matrix[[j, pivot_row]] = augmented_matrix[[pivot_row, j]]

        for i in range(j + 1, n):
            m = augmented_matrix[i, j] / augmented_matrix[j, j]
            augmented_matrix[i] = augmented_matrix[i] - m * augmented_matrix[j]
        
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            sum_ax = sum(augmented_matrix[i, k] * x[k] for k in range(i + 1, n))

            x[i] = (augmented_matrix[i, n] - sum_ax) / augmented_matrix[i, i]

        return x

result = gauss_elimination_with_pivoting(A, b)

print(result)

