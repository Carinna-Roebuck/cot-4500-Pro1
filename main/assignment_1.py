# Approximation of sqrt(2) using an iterative method

x0 = 1.5  # Initial guess
tol = 0.000001  # Tolerance
iter_count = 0  # Iteration counter
diff = x0  # Initial difference
x = x0  # Start with initial guess

print("Approximation Algorithm using root 2")
print(f"{iter_count} : {x}")

while diff >= tol:
    iter_count += 1
    y = x
    x = (x / 2) + (1 / x)  # Iterative formula
    print(f"{iter_count} : {x}")
    diff = abs(x - y)

print(f"Convergence after {iter_count} iterations")

## Bisection method using f(x)=x^3+4x^2-10=0 with accuracy 10^-3 using a=l and b=2
def bisection_method(f, left, right, tol):
    i = 0  # Iteration counter

    print("\nThe Bisection Method using using f(x)=x^3+4x^2-10=0 with accuracy 10^-3 using a=l and b=2")
    print("Iteration |   Left    |  Right    |  Midpoint  |  f(Midpoint)")
    print("--------------------------------------------------------------")
    
    while abs(right - left) > tol:
        i += 1
        p = (left + right) / 2  # Midpoint
        f_p = f(p)

        # Print the current iteration details
        print(f"{i:^9} | {left:^10.5f} | {right:^10.5f} | {p:^11.5f} | {f_p:^12.5f}")

        # Update the interval based on the sign change
        if (f(left) < 0 and f_p > 0) or (f(left) > 0 and f_p < 0):
            right = p
        else:
            left = p  

    return p, i  # Return the approximate root and iteration count

# Define the function: f(x) = x^3 + 4x^2 - 10
def func(x):
    return x**3 + 4*x**2 - 10

# Given interval [1,2] and tolerance 10^(-3)
root, iterations = bisection_method(func, left=1, right=2, tol=1e-3)

# Print final results
print("Total iterations:", iterations)


## Fixed Point Interation using the functions (a) x=g(x)=x-x>-4x2+ 10 and (b) x=g(x)= (10-x*)"*/2
print("\nFixed Point Interation using the functions (a) x=g(x)=x-x>-4x2+ 10 and (b) x=g(x)= (10-x*)*/2")
import math

# Define the fixed-point form of the function
def g(x):
    return math.sqrt((10 - x**3) / (4 * x**2)) if (10 - x**3) >= 0 and x != 0 else float('nan')

# Fixed-Point Iteration Method
def fixed_point_iteration(g, p0, tol, max_iter):
    i = 1  # Iteration counter
    result = "Failure"  # Default result

    while i <= max_iter:
        p = g(p0)  # Compute the next approximation
        
        # Check for divergence
        if math.isnan(p):  
            print("\nFAILURE: Result diverges.")
            return None
        
        # Check for convergence
        if abs(p - p0) < tol:
            result = "Success"
            print(f"\nSUCCESS: Approximate root = {p:.10f} after {i} iterations")
            break

        # Update for the next iteration
        p0 = p
        i += 1
    
    if result == "Failure":
        print("\nFAILURE: Method did not converge within the given iterations.")
    return result

# Initial approximation, tolerance, and max iterations
p0 = 1.5  # Starting guess
tol = 1e-6  # Desired tolerance (10^-6)
max_iter = 50  # Maximum iterations

# Run the fixed-point iteration method
print("Using fixed-point iteration method:")
result = fixed_point_iteration(g, p0, tol, max_iter)

import math

# Function 1: p = sqrt((10 - p0^3) / 4)
def g1(p0):
    return math.sqrt((10 - p0**3) / 4) if (10 - p0**3) >= 0 else float('nan')

# Fixed-Point Iteration Method
def fixed_point_iteration(g, p0, tol, max_iter):
    i = 1  # Iteration counter
    result = "Failure"  # Default result
    
    print("Iteration |    p    |   g(p)   |  |g(p) - p|")
    print("------------------------------------------------")
    
    while i <= max_iter:
        p = g(p0)  # Compute the next approximation
        
        # Print iteration details
        print(f"{i:^9} | {p0:^7.5f} | {p:^7.5f} | {abs(p - p0):^12.5f}")

        # Check for divergence
        if math.isnan(p):  
            print("\nResult diverges.")
            return None
        
        # Check for convergence
        if abs(p - p0) < tol:
            result = "Success"
            print(f"\nSUCCESS: Approximate root = {p:.10f} after {i} iterations")
            break

        p0 = p  # Update for the next iteration
        i += 1
    
    if result == "Failure":
        print("\nFAILURE: Method did not converge within the given iterations.")
    return result

# Initial approximation, tolerance, and max iterations
p0 = 1.5  # Starting guess
tol = 1e-6  # 10^-6 accuracy
max_iter = 50  # Maximum iterations

# Run the fixed-point method for g1
print("\nUsing g1(p) = sqrt((10 - p0^3) / 4):")
result_g1 = fixed_point_iteration(g1, p0, tol, max_iter)



## Newton-Raphson Method 
print("\nNewton-Raphson Method using f(x)=cos(x)—x=0 over the interval [ 0, 11/2 ]")
import math

# Define the function f(x) = cos(x) - x
def f(x):
    return math.cos(x) - x

# Define the derivative of the function f'(x) = -sin(x) - 1
def f_prime(x):
    return -math.sin(x) - 1

# Newton-Raphson Method
def newton_raphson(f, f_prime, pprev, tol, max_iter):
    i = 1
    result = "Failure"  # Default result

    # Iterate using the Newton-Raphson formula
    while i <= max_iter:
        # Calculate next approximation using Newton-Raphson formula
        pnext = pprev - f(pprev) / f_prime(pprev)

        # Check if the derivative is zero (which could cause division by zero)
        if f_prime(pprev) == 0:
            print("\nFAILURE: Derivative is zero, method cannot proceed.")
            return None
        
        # Print iteration details
        print(f"Iteration {i}: pprev = {pprev:.6f}, pnext = {pnext:.6f}, |pnext - pprev| = {abs(pnext - pprev):.6f}")
        
        # Check if the solution has converged
        if abs(pnext - pprev) < tol:
            result = "Success"
            print(f"\nSUCCESS: Approximate root = {pnext:.10f} after {i} iterations")
            break

        # Update the approximation for the next iteration
        pprev = pnext
        i += 1

    # If the method did not converge within the max iterations, return failure
    if result == "Failure":
        print("\nFAILURE: Method did not converge within the given iterations.")
    return result

# Initial approximation, tolerance, and max iterations
pprev = 1.0  # Initial guess (start in the middle of the interval [0, π/2])
tol = 1e-6  # 10^-6 tolerance
max_iter = 50  # Maximum number of iterations

# Run Newton-Raphson method
print("Using the Newton-Raphson method to solve f(x) = cos(x) - x:")
result = newton_raphson(f, f_prime, pprev, tol, max_iter)