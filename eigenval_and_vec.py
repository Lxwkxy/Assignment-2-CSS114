import numpy as np

# =========================================================
# Part 1: Mathematical Functions (Manual Calculation Logic)
# =========================================================

def get_det(A):
    """Function to calculate Determinant for 2x2 and 3x3 matrices"""
    n = len(A)
    if n == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    elif n == 3:
        return (A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]) - \
               (A[0][2]*A[1][1]*A[2][0] + A[0][0]*A[1][2]*A[2][1] + A[0][1]*A[1][0]*A[2][2])

def get_inverse(A):
    """Function to calculate Inverse Matrix using Adjugate (Cofactor) method"""
    n = len(A)
    det = get_det(A)
    if abs(det) < 1e-9:
        return None # Determinant is 0, inverse does not exist
        
    if n == 2:
        adj = np.array([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]])
        return adj / det
    elif n == 3:
        cofactors = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                sub_mat = [row[:j] + row[j+1:] for r, row in enumerate(A.tolist()) if r != i]
                sub_det = sub_mat[0][0]*sub_mat[1][1] - sub_mat[0][1]*sub_mat[1][0]
                cofactors[i][j] = ((-1)**(i+j)) * sub_det
        adjugate = cofactors.T
        return adjugate / det

def get_eigenvectors(A, eigenvalue, tol=1e-7):
    """Function to find eigenvectors using Row Reduction (Gaussian Elimination)"""
    n = len(A)
    M = np.copy(A)
    for i in range(n):
        M[i][i] -= eigenvalue 

    pivot_cols = []
    r = 0
    for c in range(n):
        if r >= n: break
        
        pivot_row = r
        for i in range(r + 1, n):
            if abs(M[i][c]) > abs(M[pivot_row][c]):
                pivot_row = i
                
        if abs(M[pivot_row][c]) < tol:
            continue
            
        M[[r, pivot_row]] = M[[pivot_row, r]] 
        M[r] = M[r] / M[r][c] 
        
        for i in range(n):
            if i != r:
                M[i] = M[i] - M[i][c] * M[r]
                
        pivot_cols.append(c)
        r += 1

    free_cols = [j for j in range(n) if j not in pivot_cols]
    eigenvectors = []
    
    for free_col in free_cols:
        vec = np.zeros(n)
        vec[free_col] = 1.0
        for i, p_col in enumerate(pivot_cols):
            vec[p_col] = -M[i][free_col]
        eigenvectors.append(np.round(vec, 4))
        
    return eigenvectors

# =========================================================
# Part 2: Main Program
# =========================================================
def main():
    print("=== Assignment#2 CSS114: Diagonalization ===")
    
    while True:
        try:
            n = int(input("Enter matrix size (2 or 3): "))
            if n in [2, 3]: break
            print("Only 2x2 or 3x3 matrices are supported.")
        except ValueError:
            print("Please enter a valid integer.")

    print(f"\nEnter the elements of the {n}x{n} matrix row by row (separated by space):")
    A = []
    for i in range(n):
        while True:
            row_input = input(f"Row {i+1}: ").split()
            if len(row_input) == n:
                A.append([float(val) for val in row_input])
                break
            print(f"Please enter exactly {n} numbers.")
    A = np.array(A)

    print("\n[ Input Matrix A ]")
    print(A)

    # ---------------------------------------------------------
    # 1. Eigenvalues
    # ---------------------------------------------------------
    print("\n=====================================")
    print("1. Eigenvalues")
    print("=====================================")
    if n == 2:
        trace = A[0][0] + A[1][1]
        det = get_det(A)
        roots = np.roots([1, -trace, det])
        
    elif n == 3:
        c2 = A[0][0] + A[1][1] + A[2][2] 
        
        minor_11 = (A[1][1] * A[2][2]) - (A[1][2] * A[2][1])
        minor_22 = (A[0][0] * A[2][2]) - (A[0][2] * A[2][0])
        minor_33 = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0])
        c1 = -(minor_11 + minor_22 + minor_33)
        
        c0 = get_det(A) 
        
        roots = np.roots([-1, c2, c1, c0])

    eigenvalues = eigenvalues = sorted([float(round(r.real, 4)) for r in roots if np.isreal(r)], reverse=True)
    unique_evals = list(set(eigenvalues))
    unique_evals.sort(reverse=True)
    
    print(f"Eigenvalues are: {eigenvalues}")

    # ---------------------------------------------------------
    # 2. Eigenvectors
    # ---------------------------------------------------------
    print("\n=====================================")
    print("2. Eigenvectors")
    print("=====================================")
    all_eigenvectors = []
    for ev in unique_evals:
        print(f"\n>> For eigenvalue lambda = {ev}")
        vecs = get_eigenvectors(A, ev)
        for i, v in enumerate(vecs):
            print(f"   Vector {i+1}: {v}")
            all_eigenvectors.append((ev, v))

    # ---------------------------------------------------------
    # 3., 4., 5. Diagonalization Check & Matrices P, P^-1, D
    # ---------------------------------------------------------
    print("\n=====================================")
    print("3. Diagonalization Check")
    print("=====================================")
    
    if len(all_eigenvectors) == n:
        print("Result: The matrix is diagonalizable.")
        
        print("\n=====================================")
        print("4. Matrices P, P^-1, and D")
        print("=====================================")
        
        P = np.column_stack([vec for ev, vec in all_eigenvectors])
        print("\n[ Matrix P ]")
        print(P)
        
        P_inv = get_inverse(P)
        print("\n[ Matrix P^-1 ]")
        print(np.round(P_inv, 4))
        
        D = np.diag([ev for ev, vec in all_eigenvectors])
        print("\n[ Matrix D ]")
        print(D)
        
    else:
        print("Result: The matrix is not diagonalizable.")
        print("\n=====================================")
        print("5. Reason for Non-diagonalizability")
        print("=====================================")
        print("Reason: The dimension of the eigenspace (number of linearly independent eigenvectors)")
        print("        is less than the dimension of the matrix.")
        print("        (There are not enough eigenvectors to form an n x n matrix).")

if __name__ == "__main__":
    main()