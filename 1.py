import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import lu

def f(x):
    return x**3 - x - 2

def bisection_method(a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        return None, None
    iterations = 0
    while (b - a) / 2 > tol:
        iterations += 1
        c = (a + b) / 2
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iterations

def secant_method(a, b, tol=1e-6):
    iterations = 0
    x0, x1 = a, b
    while abs(x1 - x0) > tol:
        iterations += 1
        if f(x1) - f(x0) == 0:
            return None, None
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    return x1, iterations

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def graphical_method():
    window = tk.Toplevel()
    window.title("Graphical Method and Absolute Error")
    ttk.Label(window, text="Plot f(x)=e−x−x2").pack()
    ttk.Label(window, text="Enter interval [a, b]:").pack()
    interval_entry = ttk.Entry(window)
    interval_entry.pack()
    
    ttk.Label(window, text="Enter numerical root:").pack()
    entry = ttk.Entry(window)
    entry.pack()
    
    def plot():
        try:
            a, b = map(float, interval_entry.get().split(','))
            x = np.linspace(a, b, 100)
            y = np.exp(1) - x - x**2
            plt.plot(x, y, label='f(x) = e - x - x^2')
            plt.axhline(0, color='red', linestyle='--')
            plt.legend()
            plt.show()
        except:
            messagebox.showerror("Error", "Enter a valid interval, e.g., 0,2")
    
    def calculate_error():
        try:
            numerical_root = float(entry.get())
            a, b = map(float, interval_entry.get().split(','))
            true_root = fsolve(lambda x: np.exp(1) - x - x**2, (a + b) / 2)[0]
            abs_error = abs(true_root - numerical_root)
            messagebox.showinfo("Absolute Error", f"True Root: {true_root:.6f}\nYour Root: {numerical_root:.6f}\nAbsolute Error: {abs_error:.6e}")
        except:
            messagebox.showerror("Error", "Enter valid values for interval and numerical root")
    
    ttk.Button(window, text="Plot", command=plot).pack()
    ttk.Button(window, text="Calculate Error", command=calculate_error).pack()

def compare_methods():
    window = tk.Toplevel()
    window.title("Comparison of Root-Finding Methods")
    
    ttk.Label(window, text="Enter interval [a, b]:").pack()
    interval_entry = ttk.Entry(window)
    interval_entry.pack()
    
    def calculate():
        try:
            a, b = map(float, interval_entry.get().split(','))
            root_bisect, iter_bisect = bisection_method(a, b)
            root_secant, iter_secant = secant_method(a, b)
            
            if root_bisect is None or root_secant is None:
                messagebox.showerror("Error", "Invalid interval or division by zero.")
                return
            
            true_root = root_secant
            rel_error_bisect = abs((true_root - root_bisect) / true_root) if true_root != 0 else 0
            rel_error_secant = abs((true_root - root_secant) / true_root) if true_root != 0 else 0
            
            result = (f"Bisection Method: Root = {root_bisect:.6f}, Iterations = {iter_bisect}, "
                      f"Relative Error = {rel_error_bisect:.6e}\n"
                      f"Secant Method: Root = {root_secant:.6f}, Iterations = {iter_secant}, "
                      f"Relative Error = {rel_error_secant:.6e}")
            
            messagebox.showinfo("Results", result)
        except:
            messagebox.showerror("Error", "Enter a valid interval, e.g., 1,2")
    
    ttk.Button(window, text="Calculate", command=calculate).pack()

def gauss_seidel_method():
    window = tk.Toplevel()
    window.title("Gauss-Seidel Method")
    
    matrix_entries = []
    b_entries = []
    for i in range(3):
        frame = ttk.Frame(window)
        frame.pack()

        ttk.Label(frame, text=f"Row {i+1}:").pack(side="left")
        row_entry = ttk.Entry(frame, width=15)
        row_entry.pack(side="left")

        ttk.Label(frame, text=" = ").pack(side="left")  # Равенство в той же строке

        b_entry = ttk.Entry(frame, width=5)
        b_entry.pack(side="left")

        matrix_entries.append(row_entry)
        b_entries.append(b_entry)    
    ttk.Label(window, text="Enter initial guess (comma-separated):").pack()
    x0_entry = ttk.Entry(window)
    x0_entry.pack()
    
    def solve():
        try:
            A = np.array([list(map(float, row.get().split(','))) for row in matrix_entries])
            b = np.array([float(entry.get()) for entry in b_entries])
            x0 = np.array(list(map(float, x0_entry.get().split(','))))
            x, iterations = gauss_seidel(A, b, x0)
            messagebox.showinfo("Solution", f"Solution: {x}\nIterations: {iterations}")
        except:
            messagebox.showerror("Error", "Invalid input format.")
    
    ttk.Button(window, text="Solve", command=solve).pack()

def lu_factorization():
    window = tk.Toplevel()
    window.title("LU Factorization")
    
    ttk.Label(window, text="Enter matrix (comma-separated rows):").pack()
    matrix_entry = tk.Text(window, height=5, width=40)
    matrix_entry.pack()
    
    def solve():
        try:
            matrix_lines = matrix_entry.get("1.0", "end").strip().split("\n")
            A = np.array([list(map(float, row.split(','))) for row in matrix_lines])
            P, L, U = lu(A)
            
            result = (f"L:\n{L}\n\nU:\n{U}")
            messagebox.showinfo("LU Decomposition", result)
        except:
            messagebox.showerror("Error", "Invalid matrix format. Enter rows as comma-separated values.")
    
    ttk.Button(window, text="Compute LU", command=solve).pack()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Computational Math Toolkit")
    ttk.Button(root, text="Task 1: Graphical Method", command=graphical_method).pack(pady=10)
    ttk.Button(root, text="Task 2: Root-Finding Methods", command=compare_methods).pack(pady=10)
    ttk.Button(root, text="Task 3: Gauss-Seidel Method", command=gauss_seidel_method).pack(pady=10)
    ttk.Button(root, text="Task 4: LU Factorization", command=lu_factorization).pack(pady=10)

    root.mainloop()
