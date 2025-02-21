# Characteristic functions inversion package
[![Tests](https://github.com/lilyreber/Numerical-inversion-of-characteristic-functions/actions/workflows/python-package.yml/badge.svg)](https://github.com/lilyreber/Numerical-inversion-of-characteristic-functions/actions/workflows/python-package.yml)
![GitHub License](https://img.shields.io/github/license/lilyreber/Numerical-inversion-of-characteristic-functions)


The characteristic function is the Fourier transform of a random variable distribution. One of the ways to define the distribution law is to define a characteristic function. In many probabilistic models, only the characteristic functions are available to us, and not the densities themselves, which complicates the modeling process and the evaluation of numerical characteristics. The reconstruction of the distribution function or density of a random variable by its characteristic function by analytical methods is often an extremely difficult task, therefore it is necessary to resort to the use of numerical methods.



## Installation

Clone the repository:

```bash
git clone https://github.com/lilyreber/Numerical-inversion-of-characteristic-functions.git
```

Install dependencies:

```bash
poetry install
```

## Usage example
```python
import numpy as np
import matplotlib.pyplot as plt

import CFInvert.CharFuncInverter.Bohman.BohmansInverters as bi
from CFInvert.Distributions.uniform import Unif 
from CFInvert.Standardizer import Standardizer

# Uniform distribution parameters
a = -100
b = 20
unif_distr = Unif(a, b)

# Create an array of points for plotting
t = np.linspace(-200, 200, 1000)

# Calculate the exact distribution function for uniform distribution 
unif_cdf = unif_distr.cdf(t)

# Standardization of a random variable
m = (a + b) / 2  # Expectation
var = ((b - a) ** 2) / 12  # Variance
st = Standardizer(m=m, sd=(var**0.5))

# The characteristic function of a standardized random variable
z_chr = st.standardize_chf(unif_distr.chr)

# Initialization and configuration of the inverter (Bohmann method)
inverter = bi.BohmanE(N=1e3)
inverter.fit(z_chr)

# Approximate distribution function for a standardized random variable
approx_z_cdf = inverter.cdf

# Approximate distribution function for the initial random variable
approx_cdf = st.unstandardize_cdf(approx_z_cdf)
approx_cdf_values = approx_cdf(t)

# Plotting graphs
plt.figure(figsize=(10, 6))
plt.plot(t, unif_cdf, label="The exact distribution function", linewidth=2, color="blue")
plt.plot(t, approx_cdf_values, label="Approximate distribution function", linestyle="--", linewidth=2, color="red")

plt.title("Comparison of exact and approximate distribution functions", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("F(x)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.axvline(x=a, color="green", linestyle=":", label=f"a = {a}")
plt.axvline(x=b, color="purple", linestyle=":", label=f"b = {b}")
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()
```
![example](examples/plots/uniform.png)

## Requirements

- python 3.10+
- numpy 2.2.0+
- scipy 1.15.0+
- matplotlib 3.9.3+
