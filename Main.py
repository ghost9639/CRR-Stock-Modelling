"""Binomial Model Implementations"""

# Packages ===========================================
# base
import math
import time 
from decimal import *

# need installation
import stats
from scipy.stats import norm
from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt


# Question 1 =========================================
def input_data (S_0, r, sigma, T, K, M):  
    """Validates variable entries for Binomial Model Expects initial price,
    interest rate, volatility, time till maturity, strike price, and number of
    time periods.

    If given an incorrect input, the function will print a list of all flagged
    inputs and return -1, if given a correct input, the function returns 0"""
    
    _fail = False # helper condition

    # The initial price can be any real valued number greater than 0
    if not (isinstance(S_0, (int, float, Decimal)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 

    # The interest rate is expected to be a real valued number greater than 0
    if not (isinstance(r, (int, float, Decimal)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True

    # Volatility should be entered as a +percentage (probably an integer) but
    # support is offered for floats and Decimals since theoretically fine
    if not (isinstance(sigma, (int, float, Decimal)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True

    # Our time periods only have to be positive numbers (T=1 => 1 year maturity)
    if not (isinstance(T, (int, float, Decimal)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True

    # K needs to fulfil the same requirements as S0
    if not (isinstance(K, (int, float, Decimal)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True

    # We must have at least one period or the model is redundant, it must be an integer
    if not (isinstance(M, int) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:               # only calculates when inputs are numeric, no type errors
        
        delta_t = T / M
        
        R = math.exp(r * delta_t)
        disc = math.exp(-r * delta_t)
        
        sigma = sigma / 100.0
        u = math.exp(sigma * math.sqrt(delta_t))
        d = 1.0 / u
        q = (R - d) / (u - d)

        if  q <= 0 or q >= 1:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1
    
    return 0

# Question 2 =========================================
def compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, M):
    """Basic Path Independent European Put Binomial Algorithm
    
    This basic version focuses on being clear to understand and obvious to debug
    rather than highly optimised. Expects initial price, interest rate,
    volatility, time till maturity in years, the strike price, and the number
    of periods. Expects minimal library support, only uses Math and base Python
    functionality.

    Returns -1 where inputs are incorrect, and a positive float / Decimal when
    inputs are correct."""

    _fail = False
    
    if not (isinstance(S_0, (int, float, Decimal)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 
        
    if not (isinstance(r, (int, float, Decimal)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True
        
    if not (isinstance(sigma, (int, float, Decimal)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True
        
    if not (isinstance(T, (int, float, Decimal)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True
        
    if not (isinstance(K, (int, float, Decimal)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True
        
    if not (isinstance(M, int) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:
        
        delta_t = T / M
        
        R = math.exp(r * delta_t)
        disc = math.exp(-r * delta_t)
        
        sigma = sigma / 100.0
        u = math.exp(sigma * math.sqrt(delta_t))
        d = 1.0 / u
        q = (R - d) / (u - d)

        if  q <= 0 or q >= 1:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1
    
    # We now know that there is no arbitrage and all variables meet requirements

    V = [[0.0 for _ in range(M+1)] for _ in range (M+1)] # list comprehension for speed

    for j in range (M+1):
        
        S = S_0 * (u ** j) * (d ** (M-j))
        V[j][M] = max(K - S, 0.0)

        
    for i in range (M-1, -1, -1):
        for j in range (i+1):
            V[j][i] = disc * (q * V[j+1][i+1] + (1 - q) * V[j][i+1])
            
    return V[0][0]

# Question 3 =========================================
def testing_binomial_algorithm (S_0 = 80, r = 0.025, sigma = 30, T = 1.5, K = 90, M = 50):
    """Test case for Question 3

    Expects initial price, interest rate, volatility, time till maturity, strike price, and
    iterations. Gets value from binomial function and then directly calculates put using
    Black-Scholes. Does not check for arbitrage in Black-Scholes calculation."""

    sigma = sigma / 100
    d_1 = (math.log(S_0) - math.log(K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))

    d_2 = d_1 - sigma * math.sqrt(T)

    P_t = K * math.exp(- r * T) * norm.cdf(-d_2) - S_0 * norm.cdf(-d_1)
    #     100, 0.025, 30, 1.5, 90, 200 S_0, r, sigma, T, K, M
    binom_Pt = compute_binomial_algorithm1_2 (S_0, r, sigma * 100, T, K, M)

    err = (binom_Pt - P_t) / P_t

    print(f"Price using equation: {P_t}, \nPrice using binomial model {binom_Pt} \nError: {err}%")

# Question 4 =========================================
def timing_binomial_function_1 (S_0 = 80, r = 0.025, sigma = 30, T = 1.5, K = 90):
    """Performance testing function for Question 4

    Uses the Python time library to benchmark the binomial function"""

    M = [10, 100, 1000, 10_000]
    times = []

    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times.append(time_taken)

    plt.loglog(M, times)
    plt.xlabel("M (iterations)")
    plt.ylabel("Time (seconds)")
    plt.title("Time Complexity of Base Binomial Algorithm")
    plt.show()

# Question 5 =========================================
def compute_binomial_algorithm12 (S_0, r, sigma, T, K, M):
    """Basic Path Independent European Put Binomial Algorithm
    
    This basic version focuses on being clear to understand and obvious to debug
    rather than highly optimised. Expects initial price, interest rate,
    volatility, time till maturity in years, the strike price, and the number
    of periods. Expects minimal library support, only uses Math and base Python
    functionality.

    Returns -1 where inputs are incorrect, and a positive float / Decimal when
    inputs are correct."""

    _fail = False
    
    if not (isinstance(S_0, (int, float, Decimal)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 
        
    if not (isinstance(r, (int, float, Decimal)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True
        
    if not (isinstance(sigma, (int, float, Decimal)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True
        
    if not (isinstance(T, (int, float, Decimal)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True
        
    if not (isinstance(K, (int, float, Decimal)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True
        
    if not (isinstance(M, int) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:
        
        delta_t = T / M
        
        R = math.exp(r * delta_t)
        disc = math.exp(-r * delta_t)
        
        sigma = sigma / 100.0
        u = math.exp(sigma * math.sqrt(delta_t))
        d = 1.0 / u
        q = (R - d) / (u - d)

        if  q <= 0 or q >= 1:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1
    
    # We want to use a single vector rather than a matrix

    V = [0.0 for _ in range(M+1)] # list comprehension for speed

    for j in range (M+1):
        
        S = S_0 * (u ** j) * (d ** (M-j))
        V[j] = max(K - S, 0.0)

        
    for i in range (M-1, -1, -1):
        for j in range (i + 1):
            V[j] = disc * (q * V[j+1] + (1 - q) * V[j])
            
    return V[0]

def timing_binomial_function_2 (S_0 = 80, r = 0.025, sigma = 30, T = 1.5, K = 90):
    """Performance testing function for Question 4

    Uses the Python time library to benchmark the binomial function"""

    M = [10, 100, 1000, 10_000]
    times_1 = []
    times_2 = []

    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_1.append(time_taken)
    
    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm12 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_2.append(time_taken)

    plt.loglog(M, times_1, marker = "o")
    plt.loglog(M, times_2, marker = "x")
    plt.xlabel("M (iterations)")
    plt.ylabel("Time (seconds)")
    plt.title("Time Complexity of Both Binomial Algorithms")
    plt.legend(["Base", "Low Memory"])
    plt.show()

def quick_binomial_algorithm (S_0, r, sigma, T, K, M):
    """Faster European Put Binomial Pricing Model

    Expects standard inputs. Employs numpy for speed"""

    _fail = False # helper condition

    # The initial price can be any real valued number greater than 0
    if not (isinstance(S_0, (int, float, Decimal)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 

    # The interest rate is expected to be a real valued number greater than 0
    if not (isinstance(r, (int, float, Decimal)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True

    # Volatility should be entered as a +percentage (probably an integer) but
    # support is offered for floats and Decimals since theoretically fine
    if not (isinstance(sigma, (int, float, Decimal)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True

    # Our time periods only have to be positive numbers (T=1 => 1 year maturity)
    if not (isinstance(T, (int, float, Decimal)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True

    # K needs to fulfil the same requirements as S0
    if not (isinstance(K, (int, float, Decimal)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True

    # We must have at least one period or the model is redundant, it must be an integer
    if not (isinstance(M, int) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:               # only calculates when inputs are numeric, no type errors
        
        delta_t = np.exp(np.log(T) - np.log(M))
        
        R = np.exp(r * delta_t)
        disc = np.exp(-r * delta_t)
        
        sigma = np.exp(np.log(sigma) - np.log(100))
        u = np.exp(sigma * np.sqrt(delta_t))
        d = np.exp(-sigma * np.sqrt(delta_t))
        q = (R - d) / (u - d)

        if  q <= 0.0 or q >= 1.0:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1

    j = np.arange (M+1, dtype=np.float64)
    ST = S_0 * (u ** j) * d ** (M - j)
    V = np.maximum (K - ST, 0.0)

    for i in range (M-1, -1, -1):
        V[:i+1] = disc * (q * V[1:i+2] + (1 - q) * V[:i+1])

    return V[0]

def timing_binomial_function_3 (S_0 = 80, r = 0.025, sigma = 30, T = 1.5, K = 90):
    """Performance testing function for Question 4

    Uses the Python time library to benchmark the binomial function"""

    M = [10, 100, 1000, 10_000]
    times_1 = []
    times_2 = []
    times_3 = []

    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_1.append(time_taken)
    
    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm12 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_2.append(time_taken)

    for num in M:

        start_time = time.perf_counter()
        quick_binomial_algorithm (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_3.append(time_taken)

    plt.loglog(M, times_1, marker = "o")
    plt.loglog(M, times_2, marker = "x")
    plt.loglog(M, times_3, marker = "v")
    plt.xlabel("M (iterations)")
    plt.ylabel("Time (seconds)")
    plt.title("Time Complexity of Three Binomial Algorithms")
    plt.legend(["Base", "Low Memory", "Quick"])
    plt.show()

def model_accuracies_1 ():

    known_output = [
        [100, 0.025, 30, 1.5, 90, 200, 7.951759601479267],
        [10, 0.06, 20, 1, 14, 5, 3.2719288406522917],
        [100, 0.05, 30, 0.5, 105, 1280, 9.805954112130756],
    ]

    accuracy_1 = []
    accuracy_2 = []
    accuracy_3 = []
    
    for mat in known_output:
        acc = abs((compute_binomial_algorithm1_2 (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_1.append(acc)

    for mat in known_output:
        acc = abs((compute_binomial_algorithm12 (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_2.append(acc)

    for mat in known_output:
        acc = abs((quick_binomial_algorithm (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_3.append(acc)        


    labs = ["Base", "Low Memory", "Quick"]
    vals = [np.mean(accuracy_1), np.mean(accuracy_2), np.mean(accuracy_3)]

    plt.figure(figsize = (8, 4))
    plt.bar (labs, vals)
    plt.ylabel ("Mean Testing Error (%)")
    plt.xlabel ("Model")
    plt.title ("Relative Accuracy of Models")
    plt.show()

# Question 6 =========================================

def tilted_tree (S_0, r, sigma, T, K, M):
    """Tilted Tree Binomial Model

    Expects standard inputs. Employs numpy for speed, convergence is more stable across odd
    and even iterations."""

    _fail = False # helper condition

    # The initial price can be any real valued number greater than 0
    if not (isinstance(S_0, (int, float, Decimal)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 

    # The interest rate is expected to be a real valued number greater than 0
    if not (isinstance(r, (int, float, Decimal)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True

    # Volatility should be entered as a +percentage (probably an integer) but
    # support is offered for floats and Decimals since theoretically fine
    if not (isinstance(sigma, (int, float, Decimal)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True

    # Our time periods only have to be positive numbers (T=1 => 1 year maturity)
    if not (isinstance(T, (int, float, Decimal)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True

    # K needs to fulfil the same requirements as S0
    if not (isinstance(K, (int, float, Decimal)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True

    # We must have at least one period or the model is redundant, it must be an integer
    if not (isinstance(M, int) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:               # only calculates when inputs are numeric, no type errors
        
        delta_t = np.exp(np.log(T) - np.log(M))
        
        R = np.exp(r * delta_t)
        disc = np.exp(-r * delta_t)
        
        sigma = np.exp(np.log(sigma) - np.log(100))
        u = np.exp(sigma * np.sqrt(delta_t) + 1.0 / M * np.log(K / S_0))
        d = np.exp(-sigma * np.sqrt(delta_t) + 1.0 / M * np.log(K / S_0))
        q = (R - d) / (u - d)

        if  q <= 0.0 or q >= 1.0:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1

    j = np.arange (M+1, dtype=np.float64)
    ST = S_0 * (u ** j) * d ** (M - j)
    V = np.maximum (K - ST, 0.0)

    for i in range (M-1, -1, -1):
        V[:i+1] = disc * (q * V[1:i+2] + (1 - q) * V[:i+1])

    return V[0]

def Richardson_extrapolation (S_0, r, sigma, T, K, M):
    """Applies Richardson's extrapolation using tilted trees

    Efficiently converges on an approximation in O(M^-2)"""

    P_M = tilted_tree (S_0, r, sigma, T, K, M)
    P_half = tilted_tree (S_0, r, sigma, T, K, M * 2)

    if P_M == -1 or P_half == -1:
        return -1

    else:
        return 2 * P_M - P_half

def timing_binomial_function_4 (S_0 = 80, r = 0.025, sigma = 30, T = 1.5, K = 90):
    """Performance testing function for Question 4

    Uses the Python time library to benchmark the binomial function"""

    M = [10, 100, 1000, 10_000]
    times_1 = []
    times_2 = []
    times_3 = []
    times_4 = []

    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_1.append(time_taken)
    
    for num in M:
    
        start_time = time.perf_counter()
        compute_binomial_algorithm12 (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_2.append(time_taken)

    for num in M:

        start_time = time.perf_counter()
        quick_binomial_algorithm (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_3.append(time_taken)

    for num in M:

        start_time = time.perf_counter()
        Richardson_extrapolation (S_0, r, sigma, T, K, num)
        end_time = time.perf_counter()
        time_taken = end_time - start_time

        times_4.append(time_taken)

    plt.loglog(M, times_1, marker = "o")
    plt.loglog(M, times_2, marker = "x")
    plt.loglog(M, times_3, marker = "v")
    plt.loglog(M, times_4, marker = "8")
    plt.xlabel("M (iterations)")
    plt.ylabel("Time (seconds)")
    plt.title("Time Complexity of Three Binomial Algorithms")
    plt.legend(["Base", "Low Memory", "Quick", "Richardson"])
    plt.show()

def model_accuracies_2 ():
    """Calculates and presents model accuracies"""

    known_output = [
        [100, 0.025, 30, 1.5, 90, 200, 7.951759601479267],
        [10, 0.06, 20, 1, 14, 5, 3.2719288406522917],
        [100, 0.05, 30, 0.5, 105, 1280, 9.805954112130756],
    ]

    accuracy_1 = []
    accuracy_2 = []
    accuracy_3 = []
    accuracy_4 = []
    
    for mat in known_output:
        acc = abs((compute_binomial_algorithm1_2 (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_1.append(acc)

    for mat in known_output:
        acc = abs((compute_binomial_algorithm12 (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_2.append(acc)

    for mat in known_output:
        acc = abs((quick_binomial_algorithm (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_3.append(acc)

    for mat in known_output:
        acc = abs((Richardson_extrapolation (*mat[0:6]) - mat[6]) / mat[6])
        accuracy_4.append(acc)        


    labs = ["Base", "Low Memory", "Quick", "Richardson"]
    vals = [np.mean(accuracy_1), np.mean(accuracy_2), np.mean(accuracy_3), np.mean(accuracy_4)]

    plt.figure(figsize = (8, 4))
    plt.bar (labs, vals)
    plt.ylabel ("Mean Testing Error (%)")
    plt.xlabel ("Model")
    plt.title ("Relative Accuracy of Models")
    plt.show()

if __name__ == "__main__":

    # Question 3 ===================================================
    print(compute_binomial_algorithm1_2 (80, 0.025, 30, 1.5, 90, 50)) # 15.767610655878515
    testing_binomial_algorithm()

    timing_binomial_function_1()

    print(compute_binomial_algorithm12 (80, 0.025, 30, 1.5, 90, 50))
    
    print(quick_binomial_algorithm(80, 0.025, 30, 1.5, 90, 50))
    print(tilted_tree(80, 0.025, 30, 1.5, 90, 50))

    print(Richardson_extrapolation(80, 0.025, 30, 1.5, 90, 50))

    model_accuracies_2()
    
