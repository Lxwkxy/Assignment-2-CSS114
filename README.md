# Assignment#2 CSS114: Matrix Diagonalization (Detailed Documentation)

This Python program calculates **Eigenvalues** and **Eigenvectors** to perform **Matrix Diagonalization** for 2x2 and 3x3 matrices. It is designed to demonstrate a deep understanding of Linear Algebra algorithms by implementing them from scratch without using high-level libraries like `numpy.linalg`.

## Mathematical Principles & Implementation

The program is structured into specialized functions, each representing a core step in the diagonalization process:

### 1. Determinant & Matrix Inverse
* **`get_det(A)`**: Calculates the determinant based on the matrix size:
    * **2x2**: Uses the standard $ad - bc$ formula.
    * **3x3**: Implements **Sarrus' Rule**, summing the products of diagonals to find the scalar value.
* **`get_inverse(A)`**: Implements the **Adjugate (Cofactor) Method**:
    * Computes the **Cofactor Matrix** for each element.
    * Transposes it to find the **Adjugate Matrix**.
    * Calculates $A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)$.
    * *Safety Check:* The function verifies that $\det(A) \neq 0$ to prevent errors with singular matrices.

### 2. Eigenvalues Calculation
The program solves the **Characteristic Equation** $\det(A - \lambda I) = 0$:
* **2x2 Matrix**: Uses the relationship between Trace and Determinant to form the quadratic equation $\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$.
* **3x3 Matrix**: Calculates the coefficients for a cubic polynomial using the Trace, the sum of principal minors ($M_{11}+M_{22}+M_{33}$), and the Determinant.
* **Solver**: Employs `np.roots` to find the scalar $\lambda$ values (eigenvalues).

### 3. Eigenvectors via Row Reduction
The **`get_eigenvectors(A, eigenvalue)`** function solves the linear system $(A - \lambda I)\mathbf{v} = 0$ for each $\lambda$:
* **Gaussian Elimination with Partial Pivoting**: Performs Row Reduction to transform the matrix into Echelon Form.
* **Eigenspace Extraction**: Identifies **Free Variables** and substitutes values to find the **Basis Vectors** for the eigenspace. These vectors represent the linearly independent eigenvectors.

### 4. Diagonalization Verification
* **Check**: The matrix is diagonalizable if and only if the number of linearly independent eigenvectors equals the dimension of the matrix ($n$).
* **Matrix Construction**:
    * **Matrix P (Modal Matrix)**: Formed by placing eigenvectors as columns.
    * **Matrix D (Diagonal Matrix)**: A diagonal matrix with eigenvalues on the main diagonal.
    * The program demonstrates the relationship $D = P^{-1}AP$ or $A = PDP^{-1}$.

## How to Run
1. Ensure you have NumPy installed: `pip install numpy`.
2. Run the script:
   ```bash
   python eigenval_and_vec.py