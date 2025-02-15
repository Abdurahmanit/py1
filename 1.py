import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import lu
from scipy.integrate import simpson as simps

def f(x):
    # Define the function f(x) = x^3 - x - 2
    return x**3 - x - 2

def bisection_method(a, b, tol=1e-6):
    # Check if the initial interval [a, b] contains a root
    if f(a) * f(b) >= 0:
        return None, None
    iterations = 0
    # Perform the bisection method until the tolerance is met
    while (b - a) / 2 > tol:
        iterations += 1
        # Find the midpoint of the interval
        c = (a + b) / 2
        # Check if the midpoint is a root
        if f(c) == 0:
            break
        # Update the interval based on the sign of f(c)
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    # Return the root and the number of iterations
    return (a + b) / 2, iterations

def secant_method(a, b, tol=1e-6):
    iterations = 0
    x0, x1 = a, b
    # Perform the secant method until the tolerance is met
    while abs(x1 - x0) > tol:
        iterations += 1
        # Check if the denominator is zero to avoid division by zero
        if f(x1) - f(x0) == 0:
            return None, None
        # Calculate the next approximation using the secant formula
        x_temp = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x_temp
    # Return the root and the number of iterations
    return x1, iterations


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    # Number of equations/unknowns
    n = len(A)
    # Initial guess
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        # Copy the current solution to x_new
        x_new = np.copy(x)
        for i in range(n):
            # Summation of A[i][j] * x_new[j] for j < i
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Summation of A[i][j] * x[j] for j > i
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            # Update the solution
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        # Update the solution for the next iteration
        x = x_new
    # Return the final solution and the number of iterations
    return x, max_iter

# 1 task
def graphical_method():
    # Create a new window for the graphical method
    window = tk.Toplevel()
    window.title("Graphical Method and Absolute Error")
    
    # Add labels and entry fields for the interval and numerical root
    ttk.Label(window, text="Plot f(x)=e−x−x2").pack()
    ttk.Label(window, text="Enter interval [a, b]:").pack()
    interval_entry = ttk.Entry(window)
    interval_entry.pack()
    
    ttk.Label(window, text="Enter numerical root:").pack()
    entry = ttk.Entry(window)
    entry.pack()
    
    def plot():
        try:
            # Get the interval [a, b] from the user input and convert to floats
            a, b = map(float, interval_entry.get().split(','))
            # Generate x values within the interval and calculate corresponding y values for the function
            x = np.linspace(a, b, 100)
            y = np.exp(1) - x - x**2
            # Plot the function and display the plot
            plt.plot(x, y, label='f(x) = e - x - x^2')
            plt.axhline(0, color='red', linestyle='--')
            plt.legend()
            plt.show()
        except:
            # Display an error message if the interval input is invalid
            messagebox.showerror("Error", "Enter a valid interval, e.g., 0,2")
    
    def calculate_error():
        try:
            # Get the numerical root from the user input and convert to float
            numerical_root = float(entry.get())
            a, b = map(float, interval_entry.get().split(','))
            # Calculate the true root using fsolve and the midpoint of the interval
            true_root = fsolve(lambda x: np.exp(1) - x - x**2, (a + b) / 2)[0]
            # Calculate the absolute error between the true root and the numerical root
            abs_error = abs(true_root - numerical_root)
            # Display the true root, numerical root, and absolute error in a message box
            messagebox.showinfo("Absolute Error", f"True Root: {true_root:.6f}\nYour Root: {numerical_root:.6f}\nAbsolute Error: {abs_error:.6e}")
        except:
            # Display an error message if the inputs are invalid
            messagebox.showerror("Error", "Enter valid values for interval and numerical root")
    
    # Add buttons for plotting and calculating the error
    ttk.Button(window, text="Plot", command=plot).pack()
    ttk.Button(window, text="Calculate Error", command=calculate_error).pack()

# 2 task
def compare_methods():
    # Create a new window for the comparison of root-finding methods
    window = tk.Toplevel()
    window.title("Comparison of Root-Finding Methods")
    
    # Add labels and entry fields for the interval
    ttk.Label(window, text="Find the root of f(x)=x^3−x−2 in the interval").pack()
    ttk.Label(window, text="Enter interval [a, b]:").pack()
    interval_entry = ttk.Entry(window)
    interval_entry.pack()
    
    def calculatebisect():
        try:
            # Get the interval [a, b] from the user input and convert to floats
            a, b = map(float, interval_entry.get().split(','))
            # Use the bisection method to find the root and number of iterations
            root_bisect, iter_bisect = bisection_method(a, b)
            # Use the secant method to find the root and number of iterations
            root_secant, iter_secant = secant_method(a, b)
            if root_bisect is None or root_secant is None:
                # Display an error message if the interval is invalid or division by zero occurs
                messagebox.showerror("Error", "Invalid interval or division by zero.")
                return
            
            # Calculate the true root as the average of the roots found by both methods
            true_root = (root_bisect + root_secant) / 2
            # Calculate the relative errors for both methods
            rel_error_bisect = abs((true_root - root_bisect) / true_root) if true_root != 0 else 0
            rel_error_secant = abs((true_root - root_secant) / true_root) if true_root != 0 else 0
            
            # Create a result string for the bisection method
            result = (f"Bisection Method: Root = {root_bisect:.6f}, Iterations = {iter_bisect}, "
                      f"Relative Error = {rel_error_bisect:.6e}\n")
            
            # Display the results in a message box
            messagebox.showinfo("Results", result)
        except:
            # Display an error message if the interval input is invalid
            messagebox.showerror("Error", "Enter a valid interval, e.g., 1,2")
    
    def calculatesecant():
        try:
            # Get the interval [a, b] from the user input and convert to floats
            a, b = map(float, interval_entry.get().split(','))
            # Use the bisection method to find the root and number of iterations
            root_bisect, iter_bisect = bisection_method(a, b)
            # Use the secant method to find the root and number of iterations
            root_secant, iter_secant = secant_method(a, b)
            
            if root_secant is None:
                # Display an error message if the interval is invalid or division by zero occurs
                messagebox.showerror("Error", "Invalid interval or division by zero.")
                return
            
            # Calculate the true root as the average of the roots found by both methods
            true_root = (root_bisect + root_secant) / 2
            # Calculate the relative error for the secant method
            rel_error_secant = abs((true_root - root_secant) / true_root) * 100 if true_root != 0 else 0
            
            # Create a result string for the secant method
            result = (f"Secant Method: Root = {root_secant:.6f}, Iterations = {iter_secant}, "
                      f"Relative Error = {rel_error_secant:.6e}")
            
            # Display the results in a message box
            messagebox.showinfo("Results", result)
        except:
            # Display an error message if the interval input is invalid
            messagebox.showerror("Error", "Enter a valid interval, e.g., 1,2")
    
    # Add buttons for calculating using the bisection and secant methods
    ttk.Button(window, text="Bisect", command=calculatebisect).pack()
    ttk.Button(window, text="Secant", command=calculatesecant).pack()


def gauss_seidel(A, b, x0, tol=1e-6, max_iterations=100):
    # Number of equations/unknowns
    n = len(A)
    # Initial guess
    x = x0.copy()
    
    for k in range(max_iterations):
        # Copy the current solution to x_new
        x_new = x.copy()
        
        for i in range(n):
            # Summation of A[i][j] * x_new[j] for j < i
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            # Summation of A[i][j] * x[j] for j > i
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            
            # Check if the diagonal element is zero to avoid division by zero
            if A[i][i] == 0:
                raise ValueError("Zero diagonal element detected. Cannot proceed.")
            
            # Update the solution
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1  # Return the solution and the number of iterations
        
        # Update the solution for the next iteration
        x = x_new
    
    # Raise an error if the method did not converge
    raise ValueError("Gauss-Seidel method did not converge")


# 3 task
def gauss_seidel_method():
    # Create a new window for the Gauss-Seidel method
    window = tk.Toplevel()
    window.title("Gauss-Seidel Method")
    
    # Add labels and entry fields for matrix A, vector b, and initial guess
    ttk.Label(window, text="Enter matrix A (rows as comma-separated values):").pack()
    matrix_entry = tk.Text(window, height=5, width=40)
    matrix_entry.pack()

    ttk.Label(window, text="Enter vector b (comma-separated values):").pack()
    b_entry = ttk.Entry(window, width=40)
    b_entry.pack()

    ttk.Label(window, text="Enter initial guess (comma-separated):").pack()
    x0_entry = ttk.Entry(window, width=40)
    x0_entry.pack()
    
    def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
        # Number of equations/unknowns
        n = len(A)
        # Initial guess
        x = np.array(x0, dtype=float)
        
        for k in range(max_iter):
            # Copy the current solution to x_new
            x_new = np.copy(x)
            for i in range(n):
                # Summation of A[i][j] * x_new[j] for j < i
                s1 = sum(A[i][j] * x_new[j] for j in range(i))
                # Summation of A[i][j] * x[j] for j > i
                s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
                # Update the solution
                x_new[i] = (b[i] - s1 - s2) / A[i][i]
            
            # Check for convergence
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new, k + 1  # Return the solution and the number of iterations
            # Update the solution for the next iteration
            x = x_new
        
        return x, max_iter  # Return the final solution and the maximum number of iterations

    def solve():
        try:
            # Read matrix A
            matrix_lines = matrix_entry.get("1.0", "end").strip().split("\n")
            A = np.array([list(map(float, row.split(','))) for row in matrix_lines])

            # Read vector b
            b = np.array(list(map(float, b_entry.get().split(','))))

            # Read initial guess
            x0 = np.array(list(map(float, x0_entry.get().split(','))))

            # Check dimensions
            if A.shape[0] != A.shape[1]:
                raise ValueError("Matrix A must be square (n×n).")
            if A.shape[0] != len(b):
                raise ValueError("Vector b must have the same length as matrix A.")
            if len(x0) != len(b):
                raise ValueError("Initial guess x0 must have the same length as vector b.")

            # Solve the system
            x, iterations = gauss_seidel(A, b, x0)
            messagebox.showinfo("Solution", f"Solution: {x}\nIterations: {iterations}")

        except ValueError as e:
            # Display an error message if there is a value error
            messagebox.showerror("Error", str(e))
        except Exception:
            # Display an error message if the input format is invalid
            messagebox.showerror("Error", "Invalid input format. Please check your inputs.")

    # Add a button to solve the system
    ttk.Button(window, text="Solve", command=solve).pack()


# 4 task
def lu_factorization():
    # Create a new window for LU factorization
    window = tk.Toplevel()
    window.title("LU Factorization")
    ttk.Label(window, text="Perform LU factorization on matrix").pack()

    # Add labels and entry field for matrix
    ttk.Label(window, text="Enter matrix (comma-separated rows):").pack()
    matrix_entry = tk.Text(window, height=5, width=40)
    matrix_entry.pack()
    
    def solve():
        try:
            # Read matrix A
            matrix_lines = matrix_entry.get("1.0", "end").strip().split("\n")
            A = np.array([list(map(float, row.split(','))) for row in matrix_lines])
            # Perform LU decomposition
            P, L, U = lu(A)
            
            # Display the results in a message box
            result = (f"L:\n{L}\n\nU:\n{U}")
            messagebox.showinfo("LU Decomposition", result)
        except:
            # Display an error message if the matrix input is invalid
            messagebox.showerror("Error", "Invalid matrix format. Enter rows as comma-separated values.")
    
    # Add a button to compute LU factorization
    ttk.Button(window, text="Compute LU", command=solve).pack()

# 5 task
def polynomial_curve_fitting():
    # Create a new window for polynomial curve fitting
    window = tk.Toplevel()
    window.title("Polynomial Curve Fitting")
    ttk.Label(window, text="Fit a quadratic curve to data points:").pack()
    ttk.Label(window, text="Example: x(0,1,2,3,4), y(0,1,4,9,16)").pack()
    ttk.Label(window, text="Enter data points (comma-separated x values and y values):").pack()
    x_entry = ttk.Entry(window, width=40)
    x_entry.pack()
    y_entry = ttk.Entry(window, width=40)
    y_entry.pack()
    
    def solve():
        try:
            # Read x and y values
            x_values = list(map(float, x_entry.get().split(',')))
            y_values = list(map(float, y_entry.get().split(',')))
            # Ensure x and y values have the same length
            if len(x_values) != len(y_values):
                raise ValueError("X and Y must have the same length")
            
            # Perform a quadratic fit
            coefficients = np.polyfit(x_values, y_values, 2)
            poly_eq = np.poly1d(coefficients)
            
            # Generate points for the fitted curve
            x_plot = np.linspace(min(x_values), max(x_values), 100)
            y_plot = poly_eq(x_plot)
            
            # Plot the data points and the fitted curve
            plt.scatter(x_values, y_values, color='red', label='Data Points')
            plt.plot(x_plot, y_plot, label=f'Fit: {poly_eq}', color='blue')
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Quadratic Curve Fitting")
            plt.show()
        except:
            # Display an error message if the input format is invalid
            messagebox.showerror("Error", "Invalid input format. Enter valid numeric values.")
    
    # Add a button to compute the polynomial fit
    ttk.Button(window, text="Compute Fit", command=solve).pack()

# 6 task
def lagrange_interpolation():
    # Create a new window for Lagrange interpolation
    window = tk.Toplevel()
    window.title("Lagrange Interpolation") 
    ttk.Label(window, text="Estimate f(5.5) given points:").pack()
    ttk.Label(window, text="Example: Estimate f(5.5) given points: x(5,6,7,8) y(25,36,49,64)").pack()
    ttk.Label(window, text="Enter data points (comma-separated x values and y values):").pack()
    x_entry = ttk.Entry(window, width=40)
    x_entry.pack()
    y_entry = ttk.Entry(window, width=40)
    y_entry.pack()
    
    ttk.Label(window, text="Enter x value to estimate:").pack()
    x_val_entry = ttk.Entry(window, width=20)
    x_val_entry.pack()
    
    def interpolate():
        try:
            # Read x and y values
            x_values = list(map(float, x_entry.get().split(',')))
            y_values = list(map(float, y_entry.get().split(',')))
            x_target = float(x_val_entry.get())
            
            # Ensure x and y values have the same length
            if len(x_values) != len(y_values):
                raise ValueError("X and Y must have the same length")
            
            def lagrange_basis(i, x):
                # Calculate the i-th Lagrange basis polynomial
                term = 1
                for j in range(len(x_values)):
                    if i != j:
                        term *= (x - x_values[j]) / (x_values[i] - x_values[j])
                return term
            
            # Calculate the interpolation value at x_target
            y_target = sum(y_values[i] * lagrange_basis(i, x_target) for i in range(len(x_values)))
            messagebox.showinfo("Interpolation Result", f"Estimated f({x_target}) = {y_target}")
        except:
            # Display an error message if the input format is invalid
            messagebox.showerror("Error", "Invalid input format. Enter valid numeric values.")
    
    # Add a button to compute the interpolation
    ttk.Button(window, text="Compute Interpolation", command=interpolate).pack()

# 7 task
def newtons_forward_difference():
    # Create a new window for Newton's forward difference method
    window = tk.Toplevel()
    window.title("Newton's Forward Difference - Second Derivative")
    ttk.Label(window, text="Example: Given data points x=[5,6,7,8] and y=[25,36,49,64], estimate d²y/dx² at x=1").pack()
    ttk.Label(window, text="Enter x values (comma-separated):").pack()
    x_entry = ttk.Entry(window, width=40)
    x_entry.pack()
    
    ttk.Label(window, text="Enter y values (comma-separated):").pack()
    y_entry = ttk.Entry(window, width=40)
    y_entry.pack()
    
    ttk.Label(window, text="Enter x value for second derivative:").pack()
    x_target_entry = ttk.Entry(window, width=20)
    x_target_entry.pack()
    
    def compute_second_derivative():
        try:
            # Read x and y values
            x_values = list(map(float, x_entry.get().split(',')))
            y_values = list(map(float, y_entry.get().split(',')))
            x_target = float(x_target_entry.get())
            
            # Ensure equal number of x and y values and at least 3 data points
            if len(x_values) != len(y_values) or len(x_values) < 3:
                raise ValueError("Invalid input. Ensure equal number of x and y values, at least 3 data points required.")
            
            h = x_values[1] - x_values[0]  # Step size
            n = len(x_values)
            
            # Calculate the forward difference table
            forward_diff = np.zeros((n, n))
            forward_diff[:, 0] = y_values
            
            for j in range(1, n):
                for i in range(n - j):
                    forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]
            
            # Calculate the second derivative using the forward difference table
            second_derivative = (forward_diff[0, 2]) / (h ** 2)
            messagebox.showinfo("Second Derivative Result", f"Estimated d²y/dx² at x={x_target} is {second_derivative}")
        except:
            # Display an error message if the input format is invalid
            messagebox.showerror("Error", "Invalid input format. Please enter valid numerical values.")
    
    # Add a button to compute the second derivative
    ttk.Button(window, text="Compute Second Derivative", command=compute_second_derivative).pack()

# 8 task
def simpsons_rule():
    # Create a new window for Simpson's rule integration
    window = tk.Toplevel()
    window.title("Simpson's Rule Integration")
    
    # Add label and entry field for the number of subintervals
    ttk.Label(window, text="Estimate ∫0π sin(x) dx using Simpson's rule").pack()
    ttk.Label(window, text="Enter number of subintervals (even number):").pack()
    
    entry_n = ttk.Entry(window)
    entry_n.pack()
    
    def compute_integral():
        try:
            # Read the number of subintervals
            n = int(entry_n.get())
            # Ensure the number of subintervals is positive and even
            if n % 2 != 0 or n <= 0:
                messagebox.showerror("Error", "Please enter a positive even number for subintervals.")
                return
            
            # Fixed limits of integration
            a, b = 0, np.pi
            x = np.linspace(a, b, n + 1)
            y = np.sin(x)
            # Calculate the integral using Simpson's rule
            integral = simps(y, x)
            messagebox.showinfo("Integration Result", f"Estimated Integral = {integral}")
        except ValueError:
            # Display an error message if the input is not a valid integer
            messagebox.showerror("Error", "Please enter a valid integer.")
    
    # Add a button to compute the integral
    ttk.Button(window, text="Compute Integral", command=compute_integral).pack()

if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    root.title("Computational Math Toolkit")
    
    # Add buttons for each task
    ttk.Button(root, text="Task 1: Graphical Method", command=graphical_method).pack(pady=10)
    ttk.Button(root, text="Task 2: Root-Finding Methods", command=compare_methods).pack(pady=10)
    ttk.Button(root, text="Task 3: Gauss-Seidel Method", command=gauss_seidel_method).pack(pady=10)
    ttk.Button(root, text="Task 4: LU Factorization", command=lu_factorization).pack(pady=10)
    ttk.Button(root, text="Task 5: Polynomial Curve Fitting", command=polynomial_curve_fitting).pack(pady=10)
    ttk.Button(root, text="Task 6: Lagrange Interpolation", command=lagrange_interpolation).pack(pady=10)
    ttk.Button(root, text="Task 7: Newton's Forward Difference", command=newtons_forward_difference).pack(pady=10)
    ttk.Button(root, text="Task 8: Simpson's Rule", command=simpsons_rule).pack(pady=10)

    # Run the application
    root.mainloop()