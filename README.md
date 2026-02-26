
# Table of Contents

1.  [Multiple Binomial European Put Implementations](#org4205a04)
2.  [Unit Testing, Benchmarking, and Reproducibility](#org003ba0b)



<a id="org4205a04"></a>

# Multiple Binomial European Put Implementations

1.  Basic back-calculation model,
2.  Optimised version using numpy (O(N<sup>-2</sup>)),
3.  Richardson Extrapolation.

Building up from a basic European put binomial market model, this project develops more optimised functions with increasing efficiency. Additionally, the Black-Scholes model is directly implemented for comparison, and the Richardson extrapolation is used for an efficient approximation with minimal porcelain code. Full [documentation](docs/Assessment1.pdf) of the process available.

![img](docs/images/Figure_7.png)

![img](docs/images/Figure_8.png)


<a id="org003ba0b"></a>

# Unit Testing, Benchmarking, and Reproducibility

Extensive use of [pytest](testing.py) to ensure functions are robust to internal alterations and changes to input validation logic. Additionally used to validate approximations remaining within reasonable bounds. Use of Python time library to benchmark functions. Nix flake including pytest attached for easy reproducibility.

