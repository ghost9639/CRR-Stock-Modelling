import pytest

from Main import input_data
from Main import compute_binomial_algorithm1_2

invalid_inputs = [
    ["a", 2, 4, 5, 2, 1], # -1 initial stock can't be letter
    [3, 'a', 4, 5, 2, 1], # -1 r can't be a letter
    [3, 4, "a", 5, 2, 1],  # -1 volatility can't be a letter
    [3, 4, 5, 'a', 2, 1],  # -1 time periods can't be a letter
    [3, 4, 5, 2, 'a', 1],  # -1 K can't be a letter
    [3, 4, 5, 2, 1, 'a'],  # -1 M can't be a letter
    [0, 5, 4, 5, 2, 1],  # -1 S_0 must be positive
    [4, -5, 4, 5, 2, 1], # -1 interest rate must be positive
    [5, 4, -1, 5, 2, 1],  # -1 sigma must be positive integer
    [5, 4, 5, -4, 2, 1],  # -1 Time period must be positive
    [4, 5, 4, 5, True, 1],   # -1 strike price cannot be a boolean
    [5, 4, 5, 2, 1, 0.99], # -1 must be at least one period
]

arbitrage_cases = [
    [10, 1.3, 0.2, 2, 12, 10],
]

valid_inputs = [[1, 0.06, 20, 0.33333, 2, 10],
                [10, 0.09, 30, 2, 14, 20]]

# S_o K r sigma T M
# 100 90 0.025 0.3 1.5 200

# 21.5ish
known_output = [
    [100, 0.025, 0.3, 1.5, 90, 200]
]

# input_data (S_0, r, sigma, T, K, M)
@pytest.mark.parametrize("args", invalid_inputs)
def test_1(args):
    assert input_data (*args) == -1

@pytest.mark.parametrize("args", valid_inputs)
def test_2(args):
    assert input_data (*args) == 0

@pytest.mark.parametrize ("args", invalid_inputs)
def test_3(args):
    assert compute_binomial_algorithm1_2(*args) == -1

#compute_binomial_algorithm1_2 (S_0, r, sigma, T, K, M)    
@pytest.mark.parametrize("args", arbitrage_cases)
def test_4 (args):
    assert compute_binomial_algorithm1_2(*args) == -1
    assert input_data(*args) == -1
    
# Temporary Test    
@pytest.mark.parametrize ("args", valid_inputs)
def test_5 (args):
    assert compute_binomial_algorithm1_2 (*args) == 0

@pytest.mark.paramtetrize ("args", known_output)
def test_6 (args):
    assert compute_binomial_algorithm1_2(*args)<23
    assert compute_binomial_algorithm1_2(*args)>20
    


