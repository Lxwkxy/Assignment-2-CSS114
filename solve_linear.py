import numpy as np
import sys

class UltimatePivotingSolver:
    def __init__(self):
        self.n = 0
        self.A = None
        self.B = None
        self.has_data = False

    def print_matrix(self, matrix, title="Matrix", highlight_row=None):
        """ แสดง Matrix พร้อมลูกศรชี้แถวที่กำลังทำงาน """
        print(f"\n   --- {title} ---")
        rows, cols = matrix.shape
        for i in range(rows):
            prefix = "-> " if (highlight_row is not None and i == highlight_row) else "   "
            row_str = prefix + "| "
            for j in range(cols):
                val = matrix[i][j]
                if abs(val) < 1e-9: val = 0.0
                row_str += f"{val:>10.4f} "
            row_str += "|"
            print(row_str)
        print()

    def input_new_data(self):
        print("\n" + "="*60)
        print("   INPUT DATA")
        print("="*60)
        try:
            print("Step 1: Enter size (n)")
            self.n = int(input(">> n = "))

            print(f"\nStep 2: Enter Matrix A ({self.n}x{self.n}) row by row.")
            matrix_a = []
            for i in range(self.n):
                while True:
                    try:
                        line = input(f"   Row {i+1}: ")
                        row = list(map(float, line.split()))
                        if len(row) != self.n:
                            print(f"   [Error] Need {self.n} numbers.")
                            continue
                        matrix_a.append(row)
                        break
                    except ValueError:
                        print("   [Error] Numbers only.")
            self.A = np.array(matrix_a, dtype=float)

            print(f"\nStep 3: Enter Vector B.")
            print("   (Enter random numbers if you only want to find Inverse)")
            while True:
                try:
                    line = input(f"   Vector B: ")
                    vector_b = list(map(float, line.split()))
                    if len(vector_b) != self.n:
                        print(f"   [Error] Need {self.n} numbers.")
                        continue
                    self.B = np.array(vector_b, dtype=float)
                    break
                except ValueError:
                     print("   [Error] Numbers only.")
            
            self.has_data = True
            print("\n   [✓] Data Saved!")

        except Exception as e:
            print(f"Error: {e}")
            self.has_data = False

    def check_system_status(self):
        if not self.has_data: return "no_data"
        det_A = np.linalg.det(self.A)
        if abs(det_A) > 1e-9:
            return "unique"
        else:
            rank_A = np.linalg.matrix_rank(self.A)
            augmented = np.column_stack((self.A, self.B))
            rank_Aug = np.linalg.matrix_rank(augmented)
            if rank_A < rank_Aug: return "no_solution"
            else: return "infinite"

    def show_final_answer(self, x):
        print("\n" + "-"*30)
        print("   FINAL SOLUTION")
        for i in range(len(x)):
            print(f"   x{i+1} = {x[i]:.4f}")
        print("-" * 30 + "\n")

    # ==========================================
    # 1. Gaussian Elimination (Pivoting + Row-by-Row)
    # ==========================================
    def solve_gauss_elimination(self):
        print("\n" + "="*70)
        print("   METHOD: GAUSSIAN ELIMINATION (With Partial Pivoting)")
        print("="*70)
        
        M = np.column_stack((self.A.copy(), self.B.copy()))
        n = self.n
        self.print_matrix(M, "Initial Augmented Matrix")

        # --- Forward Elimination ---
        print(">>> PHASE 1: Forward Elimination")
        
        for i in range(n):
            print(f"\n[Processing Column {i+1}]")
            
            # --- Pivoting ---
            pivot_idx = i + np.argmax(np.abs(M[i:, i]))
            max_val = M[pivot_idx, i]
            
            if abs(max_val) < 1e-9:
                print(f"   ! Max pivot is 0. Column is zero. Skipping.")
                continue

            if pivot_idx != i:
                print(f"   > PIVOTING: Swap Row {i+1} <-> Row {pivot_idx+1}")
                print(f"     (Moving max value {max_val:.4f} to diagonal)")
                M[[i, pivot_idx]] = M[[pivot_idx, i]]
                self.print_matrix(M, f"Matrix after Swap")
            else:
                print(f"   > Pivoting: Row {i+1} already has max value ({max_val:.4f}). No swap needed.")

            pivot_val = M[i, i]
            
            # Elimination
            elimination_happened = False
            for k in range(i+1, n):
                target_val = M[k, i]
                if abs(target_val) > 1e-9:
                    elimination_happened = True
                    factor = target_val / pivot_val
                    
                    print(f"\n   --- Eliminating A[{k+1},{i+1}] (Value: {target_val:.4f}) ---")
                    print(f"   Multiplier = {target_val:.4f} / {pivot_val:.4f} = {factor:.4f}")
                    print(f"   Operation: R{k+1} = R{k+1} - ({factor:.4f} * R{i+1})")
                    
                    M[k, i:] -= factor * M[i, i:]
                    self.print_matrix(M, f"Matrix after updating Row {k+1}", highlight_row=k)
            
            if not elimination_happened and i < n-1:
                print("   (Zeros already below pivot)")

        # --- Back Substitution ---
        print("\n>>> PHASE 2: Back Substitution")
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            rhs = M[i, n]
            lhs_coeff = M[i, i]
            
            if abs(lhs_coeff) < 1e-9: continue

            known_sum = sum(M[i, j] * x[j] for j in range(i+1, n))
            explanation_terms = [f"({M[i, j]:.2f}*{x[j]:.2f})" for j in range(i+1, n)]
            
            term_str = " + ".join(explanation_terms) if explanation_terms else "0"
            print(f"\n   Solving x{i+1}:  {lhs_coeff:.2f}*x{i+1} + [{term_str}] = {rhs:.2f}")
            
            x[i] = (rhs - known_sum) / lhs_coeff
            print(f"     x{i+1} = ({rhs:.4f} - {known_sum:.4f}) / {lhs_coeff:.4f} = {x[i]:.4f}")
        
        self.show_final_answer(x)

    # ==========================================
    # 2. Gauss-Jordan Elimination (Pivoting + Row-by-Row)
    # ==========================================
    def solve_gauss_jordan(self):
        print("\n" + "="*70)
        print("   METHOD: GAUSS-JORDAN ELIMINATION (With Partial Pivoting)")
        print("="*70)
        
        M = np.column_stack((self.A.copy(), self.B.copy()))
        n = self.n
        self.print_matrix(M, "Start")
        
        for i in range(n):
            print(f"\n[Processing Row/Col {i+1}]")
            
            # --- Pivoting ---
            pivot_idx = i + np.argmax(np.abs(M[i:, i]))
            max_val = M[pivot_idx, i]
            
            if abs(max_val) < 1e-9: continue

            if pivot_idx != i:
                print(f"   > PIVOTING: Swap Row {i+1} <-> Row {pivot_idx+1}")
                print(f"     (Moving max value {max_val:.4f} to diagonal)")
                M[[i, pivot_idx]] = M[[pivot_idx, i]]
                self.print_matrix(M, f"Matrix after Swap")
            
            # Normalize
            pivot_val = M[i, i]
            print(f"   > Normalizing Row {i+1} (Divide by {pivot_val:.4f})")
            M[i] = M[i] / pivot_val
            self.print_matrix(M, f"Matrix after Normalizing Row {i+1}", highlight_row=i)
            
            # Eliminate all other rows
            print(f"   > Eliminating other rows in Col {i+1}...")
            for k in range(n):
                if k != i:
                    factor = M[k, i]
                    if abs(factor) > 1e-9:
                        print(f"     --- Updating Row {k+1} (Target: {factor:.4f}) ---")
                        print(f"     Operation: R{k+1} = R{k+1} - ({factor:.4f} * R{i+1})")
                        M[k] -= factor * M[i]
                        self.print_matrix(M, f"Matrix after updating Row {k+1}", highlight_row=k)
        
        self.show_final_answer(M[:, n])

    # ==========================================
    # 3. LU Factorization (PA = LU with Pivoting)
    # ==========================================
    def solve_lu_factorization(self):
        print("\n" + "="*70)
        print("   METHOD: LU FACTORIZATION (With Partial Pivoting)")
        print("   Theory: PA = L * U")
        print("   (Since we use pivoting, we create a Permutation Matrix P)")
        print("="*70)
        
        status = self.check_system_status()
        if status != "unique":
            print(f"   [!] System is {status}. LU requires Non-Singular matrix.")
            return

        n = self.n
        U = self.A.copy() # Start U as A, then eliminate
        L = np.eye(n)     # Start L as Identity
        P = np.eye(n)     # Permutation Matrix
        
        print(">>> Step 1: Decomposing (PA = LU)")
        self.print_matrix(U, "Initial Matrix (U starts as A)")

        for i in range(n):
            print(f"\n   --- Processing Column {i+1} ---")
            
            # 1. Pivoting
            pivot_row = i + np.argmax(np.abs(U[i:, i]))
            max_val = U[pivot_row, i]
            
            if pivot_row != i:
                print(f"   > PIVOTING: Swap Row {i+1} <-> Row {pivot_row+1}")
                print(f"     (Because |{max_val:.4f}| > |{U[i,i]:.4f}|)")
                
                # Swap U
                U[[i, pivot_row]] = U[[pivot_row, i]]
                
                # Swap P (Track permutation)
                P[[i, pivot_row]] = P[[pivot_row, i]]
                
                # Swap L (IMPORTANT: Only swap the part of L we already calculated!)
                # i.e., columns 0 to i-1
                if i > 0:
                    print(f"     (Swapping corresponding rows in L up to col {i})")
                    L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
                
                self.print_matrix(U, "U after Swap")
                self.print_matrix(P, "P after Swap")
            
            # 2. Elimination
            pivot_val = U[i, i]
            if abs(pivot_val) < 1e-9:
                print("   [!] Pivot is zero. Singular matrix.")
                return

            for k in range(i+1, n):
                target_val = U[k, i]
                if abs(target_val) > 1e-9:
                    multiplier = target_val / pivot_val
                    
                    # Store multiplier in L
                    L[k, i] = multiplier
                    
                    print(f"\n     Eliminating U[{k+1},{i+1}] (Val: {target_val:.4f})")
                    print(f"     Multiplier (stored in L[{k+1},{i+1}]) = {multiplier:.4f}")
                    print(f"     U_Row{k+1} = U_Row{k+1} - ({multiplier:.4f} * U_Row{i+1})")
                    
                    # Update U
                    U[k, i:] -= multiplier * U[i, i:]
                    self.print_matrix(U, f"U after updating Row {k+1}", highlight_row=k)

        print("\n>>> Decomposition Result:")
        self.print_matrix(P, "P (Permutation Matrix)")
        self.print_matrix(L, "L (Lower Triangular)")
        self.print_matrix(U, "U (Upper Triangular)")
        
        # Verify PA = LU
        print("\n   [Verification] Check if P*A == L*U")
        PA = np.dot(P, self.A)
        LU = np.dot(L, U)
        self.print_matrix(PA, "P * A")
        self.print_matrix(LU, "L * U (Should match P*A)")
        
        # Solving
        # System is Ax = B
        # PAx = PB
        # LUx = PB
        # Let Ux = y, then Ly = PB
        
        print("\n>>> Step 2: Calculate New RHS (Pb)")
        Pb = np.dot(P, self.B)
        print(f"   Original B: {self.B}")
        print(f"   New B (after permuting): {Pb}")

        print("\n>>> Step 3: Forward Substitution (Ly = Pb)")
        y = np.zeros(n)
        for i in range(n):
            sum_val = sum(L[i][j] * y[j] for j in range(i))
            y[i] = Pb[i] - sum_val
            
            term_str = " - ".join([f"({L[i][j]:.2f}*{y[j]:.2f})" for j in range(i)]) if i > 0 else "0"
            print(f"   y{i+1} = {Pb[i]:.2f} - [{term_str}] = {y[i]:.4f}")

        print("\n>>> Step 4: Back Substitution (Ux = y)")
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            sum_val = sum(U[i][j] * x[j] for j in range(i+1, n))
            x[i] = (y[i] - sum_val) / U[i][i]
            
            term_str = " - ".join([f"({U[i][j]:.2f}*{x[j]:.2f})" for j in range(i+1, n)]) if i < n-1 else "0"
            print(f"   x{i+1} = ({y[i]:.4f} - [{term_str}]) / {U[i][i]:.4f} = {x[i]:.4f}")
            
        self.show_final_answer(x)

    # ==========================================
    # 4. Find Inverse Only (Cofactor/Adjoint Detail)
    # ==========================================
    def find_inverse_only(self):
        print("\n" + "="*70)
        print("   METHOD: FIND INVERSE (Adjoint Method - Step-by-Step)")
        print("   Theory: A^-1 = (1/det(A)) * adj(A)")
        print("="*70)
        
        # 1. Determinant
        det_A = np.linalg.det(self.A)
        print(f"\n1. Calculate Determinant")
        print(f"   det(A) = {det_A:.4f}")
        if abs(det_A) < 1e-9:
            print("   [!] Singular Matrix. No Inverse.")
            return

        n = self.n
        C = np.zeros((n, n))
        print(f"\n2. Matrix of Cofactors (C)")
        print("   C_ij = (-1)^(i+j) * det(Minor_ij)")
        
        for i in range(n):
            for j in range(n):
                # สร้าง Submatrix
                sub_matrix = np.delete(np.delete(self.A, i, 0), j, 1)
                minor = np.linalg.det(sub_matrix)
                sign = (-1)**(i+j)
                val = sign * minor
                C[i][j] = val
                
                # Detail print (แสดงทุกตัว!)
                print(f"\n   [Element {i+1},{j+1}]")
                print(f"     Submatrix (cut row {i+1}, col {j+1}):")
                for row_sub in sub_matrix:
                    print(f"       {row_sub}")
                
                sign_sym = "(+1)" if sign > 0 else "(-1)"
                print(f"     Minor (det) = {minor:.4f}")
                print(f"     Cofactor    = {sign_sym} * {minor:.4f} = {val:.4f}")

        print("\n3. Cofactor Matrix Result (C):")
        self.print_matrix(C, "C")
        
        adj_A = C.T
        print("4. Adjoint Matrix (Transpose of C):")
        print("   Swap rows to columns...")
        self.print_matrix(adj_A, "adj(A)")
        
        print(f"5. Final Inverse (adj(A) / det(A)):")
        print(f"   Divide every element by {det_A:.4f}")
        A_inv = adj_A / det_A
        self.print_matrix(A_inv, "A^-1")

    # ==========================================
    # 5. Solve via Inverse
    # ==========================================
    def solve_inverse_system(self):
        print("\n" + "="*70)
        print("   METHOD: SOLVE X = A^-1 * B")
        print("="*70)
        
        status = self.check_system_status()
        if status != "unique":
             print(f"   [!] Method Failed: Matrix is {status}.")
             return

        A_inv = np.linalg.inv(self.A.copy())
        self.print_matrix(A_inv, "A^-1")
        
        print("   Multiplying A^-1 * B:")
        X = np.zeros(self.n)
        for i in range(self.n):
            terms = []
            val = 0
            for j in range(self.n):
                term = A_inv[i][j] * self.B[j]
                val += term
                terms.append(f"({A_inv[i][j]:.2f}*{self.B[j]:.2f})")
            
            print(f"   x{i+1} = {' + '.join(terms)} = {val:.4f}")
            X[i] = val
            
        self.show_final_answer(X)

    # ==========================================
    # MAIN RUN
    # ==========================================
    def run(self):
        while True:
            print("\n" + "="*60)
            print("   ULTIMATE PIVOTING LINEAR SOLVER")
            print("="*60)
            if self.has_data:
                print(f"   [Current Data]: Matrix A ({self.n}x{self.n}) Loaded.")
            else:
                print("   [Current Data]: (Empty)")
            print("-" * 60)
            
            print("1. Input New Data")
            print("-" * 30)
            print("2. Gaussian Elimination (Pivoting + Step-by-Step)")
            print("3. Gauss-Jordan Elimination (Pivoting + Step-by-Step)")
            print("4. LU Factorization (PA=LU + Step-by-Step)")
            print("5. Solve using Inverse (X = A^-1 * B)")
            print("6. Find Inverse Matrix Only (Cofactor/Adj Detail)")
            print("-" * 30)
            print("0. Exit")
            
            choice = input("\nSelect Option: ")

            if choice == '1':
                self.input_new_data()
            elif choice == '0':
                print("Goodbye!")
                break
            elif choice in ['2', '3', '4', '5', '6']:
                if not self.has_data:
                    print("\n[!] No data. Select Option 1 first.")
                    continue
                
                # Check Dimensions
                if choice != '6': 
                    if self.A.shape[1] != self.B.shape[0]:
                        print("\n[!] Dimension Mismatch.")
                        continue

                if choice == '2': self.solve_gauss_elimination()
                elif choice == '3': self.solve_gauss_jordan()
                elif choice == '4': self.solve_lu_factorization()
                elif choice == '5': self.solve_inverse_system()
                elif choice == '6': self.find_inverse_only()
            else:
                print("[!] Invalid Choice")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    solver = UltimatePivotingSolver()
    solver.run()