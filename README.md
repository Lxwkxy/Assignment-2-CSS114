# Assignment#2 CSS114: Matrix Diagonalization (Detailed Documentation)

This Python program calculates **Eigenvalues** and **Eigenvectors** to perform **Matrix Diagonalization** for 2x2 and 3x3 matrices. It is designed to demonstrate a deep understanding of Linear Algebra algorithms by implementing them from scratch without using high-level libraries like `numpy.linalg`.

---

## Calculation Flow & Input Method

The program executes through a structured step-by-step pipeline, transforming raw user input into verified diagonalization matrices:

```text
[1. User Input] ➔ [2. Determinant Check] ➔ [3. Eigenvalues (np.roots)] ➔ [4. Eigenvectors (Gaussian Elimination)] ➔ [5. Diagonalization Check & Output]
```

1. **Matrix Dimension Input:** The program prompts the user to define the matrix size ($n \times n$), strictly supporting either `2` or `3`.
2. **Matrix Elements Input:** The user enters the matrix row by row. Elements within the same row must be separated by spaces (e.g., `4 1`).
3. **Determinant Evaluation:** The system computes $\det(A)$ to verify if the matrix is invertible, which is critical for the later stage of finding $P^{-1}$.
4. **Eigenvalues Derivation:** Coefficients of the characteristic polynomial are generated using matrix invariants (Trace and Determinant). These coefficients are passed to `np.roots` to solve for $\lambda$.
5. **Eigenvectors Row Reduction:** For each unique $\lambda$, the linear system $(A - \lambda I)\mathbf{v} = 0$ is constructed. The program applies **Gaussian Elimination with Partial Pivoting** to find the row-echelon form and extract the basis vectors (eigenvectors) from the free variables.
6. **Diagonalization Check & Output:** The program counts the total number of linearly independent eigenvectors. If the count equals $n$, it constructs Matrix $P$ and Matrix $D$, verifying $D = P^{-1}AP$. If not, it safely handles the exception and explains why the matrix is defective.

---

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

---

---

## Example Test Cases

You can use these test cases to verify the correctness of the program for both diagonalizable and non-diagonalizable matrices.

### Case 1: Diagonalizable Matrix (2x2)
* **Description:** A matrix with distinct real eigenvalues, which guarantees it is diagonalizable.
* **Input Matrix A:**
  ```text
  4 1
  2 3
  ```
* **Expected Output:**
  * **Eigenvalues (λ):** 5.0 and 2.0
  * **Eigenvectors:**
    * For λ = 5: [1, 1] (or any scalar multiple)
    * For λ = 2: [-0.5, 1] or [1, -2]
  * **Diagonalization Status:** Matrix is Diagonalizable
  * **Matrices Created:**
    * **Matrix P (Modal Matrix)**
    * **Matrix D (Diagonal Matrix):** 
    ```text
      [5  0]
      [0  2]
      ```
  * **Verification:** Checks if P^-1 * A * P = D holds true within a small precision tolerance.

### Case 2: Non-Diagonalizable (Defective) Matrix (2x2)
* **Description:** A matrix that has repeated eigenvalues but lacks enough linearly independent eigenvectors to form the basis (Geometric Multiplicity < Algebraic Multiplicity).
* **Input Matrix A:**
  ```text
  3 1
  0 3
  ```
* **Expected Output:**
  * **Eigenvalues (λ):** 3.0 and 3.0 (Repeated root)
  * **Eigenvectors:** Only one linearly independent eigenvector can be found: [1, 0]
  * **Diagonalization Status:** Matrix is NOT Diagonalizable (Defective Matrix)
  * **Reasoning:** The number of linearly independent eigenvectors (1) is less than the dimension of the matrix (2). The program safely terminates the diagonalization process and displays an explanation.

---

---

## How to Run
1. Ensure you have NumPy installed: `pip install numpy`
2. Run the script:
   ```bash
   python eigenval_and_vec.py
   ```
3. Input the matrix size (2 or 3).
4. Enter the matrix elements row by row, separated by spaces.