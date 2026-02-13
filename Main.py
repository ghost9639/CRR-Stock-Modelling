"""Binomial Model Implementations"""

import math
from decimal import *
import stats
import numpy as np

def input_data (S_0, r, sigma, T, K, M):  
    """Validates variable entries for Binomial Model
    Expects initial price, interest rate, volatility,
    time till maturity, strike price, and number of
    time periods."""
    _fail = False
    
    # if not (isinstance(S_0, (int, float)) and not isinstance(S_0, bool)) or (S_0 <= 0):
    #     print ("Initial stock price must take a positive numerical value,")
    #     _fail = True 
        
    if not (isinstance(r, (int, float)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True
        
    if not (isinstance(sigma, (int, float)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True
        
    if not (isinstance(T, (int, float)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True
        
    if not (isinstance(K, (int, float)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True
        
    if not (isinstance(M, (int, float)) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1

    # Now we test for arbitrage
    # need to start by calculating the single period RNMs
    delta_t = math.exp(math.log(T) - math.log(M))
    cont_rate = math.exp(r * delta_t)

    u = 1 + sigma
    d = 1 - sigma 

    if  cont_rate <= d or cont_rate >= u:
        print("Model has arbitrage")
        _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1
    
    return 0

# S_o K r sigma T M
# 100 90 0.025 0.3 1.5 200

# 21.5ish

# S = np.zeros(M+1)
# V = np.zeros(M+1)
# j=np.arrange(M+1)
# ST = S_0*u**j*d**(M-j)
# V = np.maximum(ST-K, 0)
# V[ST>=B]=0

# for i in range (M-1, -1, -1):
#   j = np.arrange(i+1)
#   ST = S_0*u**j*d**(i-j)
#   V = np.exp(-r*dt)*(q*V[1:]+(1-q)*V[:-1])
#   return V[0]

def compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, M):
    """Basic Path Independent European Put Binomial Algorithm"""

    _fail = False
    
    if not (isinstance(S_0, (int, float)) and not isinstance(S_0, bool)) or (S_0 <= 0):
        print ("Initial stock price must take a positive numerical value,")
        _fail = True 
        
    if not (isinstance(r, (int, float)) and not isinstance(r, bool)) or (r <= 0):
        print ("Interest rate must take a positive numerical value,")
        _fail = True
        
    if not (isinstance(sigma, (int, float)) and not isinstance(sigma, bool)) or (sigma <= 0):
        print ("Volatility can only be a positive numerical,")
        _fail = True
        
    if not (isinstance(T, (int, float)) and not isinstance(T, bool)) or (T <= 0):
        print ("Time period must be some positive multiple of 1,")
        _fail = True
        
    if not (isinstance(K, (int, float)) and not isinstance(K, bool)) or (K <= 0):
        print ("An option can only have a positive numerical value,")
        _fail = True
        
    if not (isinstance(M, (int, float)) and not isinstance(M, bool)) or (M < 1):
        print ("Must be at least one time period,")
        _fail = True

    if not _fail:
        
        delta_t = T / M
        cont_rate = math.exp(r * delta_t)
        sigma = sigma / 100
        u = 1 + sigma
        d = 1 - sigma
        q = (cont_rate - d) / (u - d)
        anti_q = 1 - q

        if  q <= 0 or anti_q <= 0:
            print("Model has arbitrage,")
            _fail = True 
        
    if _fail:
        print("Invalid Entry, please amend arguments.")
        return -1
    
    # We now know that there is no arbitrage and all variables meet requirements

    V = [[0 for i in range(M)] for j in range (M)] # list comprehension for speed

    for j in range (M):
        
        S = u**j * d**(M-j) * S_0
        
        if K > S:
            V[j][M-1] = K - S
        else:
            V[j][M-1] = 0

    for i in range (M-2, -1, -1):
        for j in range (i):
            V[j][i] = math.exp(-r * delta_t) * (q * V[j+1][i+1] + (1-q) * V[j][i+1])

    for i in range (M-1, -1, -1):
        print(V[i][2])
    return(V[0][0])            # since put options >0 at base of tree


if __name__ == "__main__":

    print(compute_binomial_algorithm1_2(100, 0.025, 0.3, 1.5, 90, 200))
    print(compute_binomial_algorithm1_2(10, 0.06, 20, 1, 14, 5))
     
    # compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, M)

    test_mat = [[i for i in range (3)] for j in range (4)]
    print(test_mat)

    print(test_mat[3][2])
    
    print("yes") if 0==0 else print("no")

    Decimal(1) / Decimal(7)
    
    for i in range (4, 0, -1):
        print(i)

        
