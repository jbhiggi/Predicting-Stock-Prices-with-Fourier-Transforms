# Predicting Stock Prices with Fourier Transforms

In this notebook, we explore a signal‑processing approach to short‑term stock price forecasting using the Fast Fourier Transform (FFT). By decomposing the historical price series into its constituent sinusoidal modes, we can identify and isolate the dominant cyclic components, remove noise and trends, and then analytically extend each mode beyond the observed window. Summing these extended sinusoids yields a periodic “forecast” that captures the principal oscillatory behavior in the data.

This repo is part of a growing portfolio of work related to finance. I am currently an Astrophysics PhD candidate studying in the US and wish to apply the advanced statistical analysis skills I am acquiring in observational astrophysics to financial markets.

Feel free to reach out at jbhiggi3@ncsu.edu or open an issue here on GitHub.



## Dependencies

This project relies on the following Python packages and modules:

- **pandas**  
- **numpy**  
- **matplotlib**  
- **os** (built‑in; no install required)

You can install the third‑party packages with:

```bash
pip install pandas numpy matplotlib
