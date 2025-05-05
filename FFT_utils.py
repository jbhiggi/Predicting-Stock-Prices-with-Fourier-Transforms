import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import matplotlib.pyplot as plt
import json

def load_stocks_from_hdf(hdf_filename, symbol, days=365):
    """
    Load stock data for a specific symbol from an HDF5 file and return it as a DataFrame.

    This function opens the specified HDF5 file in read-only mode, retrieves the dataset 
    associated with the given stock symbol, and processes the data by:
      - Limiting the dataset to the first 365 rows (days).
      - Resetting the index.
      - Reversing the order of the rows so that the data is in chronological order 
        (with the earliest date first).

    Parameters
    ----------
    hdf_filename : str
        The file path to the HDF5 file containing stock datasets.
    symbol : str
        The key (stock symbol) under which the desired dataset is stored in the HDF5 file.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the processed stock data for the specified symbol. 
        The data is limited to 365 days and arranged in chronological order.

    Notes
    -----
    - If the specified symbol is not found in the HDF5 file, a warning message is printed.
    - It is assumed that the dataset stored under each symbol initially has the latest date at the top.
    """

    # Check if the file exists
    if not os.path.exists(hdf_filename):
        raise FileNotFoundError(f"Error: The file '{hdf_filename}' does not exist.")
     
    stock_dict = {}  # Dictionary to store stock data
    
    with pd.HDFStore(hdf_filename, mode='r') as store:
        try:
            stock_dict[symbol] = store[symbol]  # Load the dataset for the symbol
            print(f"Loaded {symbol} data from {hdf_filename}")
        except KeyError:
            print(f"Warning: {symbol} not found in {hdf_filename}")

    df = stock_dict[symbol]

    # Ensure the dataset is not empty
    if df.empty:
        raise RuntimeError(f"Error: The dataset for '{symbol}' is empty.")

    # Limit the dataset to the specified number of days, ensuring at least `days` rows exist
    if len(df) < days:
        print(f"Warning: '{symbol}' has only {len(df)} rows, less than the requested {days}. Using all available data.")
        days = len(df)  # Use whatever is available instead of forcing `days`

    # Limit the dataset to the specified number of days
    df = df.iloc[:days].reset_index(drop=True)
    df = df.iloc[::-1].reset_index(drop=True)  # Reverse the order to chronological order

    return df

def plot_price(df, symbol, date_column='timestamp', display=True, outfile=None):
    """
     Plots the closing price.

    """
    # Ensure date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    df['SMA_20'] =  df['close'].rolling(window=20).mean()

    # Create figure-
    plt.figure(figsize=(12, 7.2))

    # Plot close price and SMA
    plt.plot(df[date_column], df['close'], label='Close Price', color='black', linewidth=1)
    plt.plot(df[date_column], df['SMA_20'], label='SMA_20', color='blue', linewidth=1)

    # Improve x-axis readability by displaying only some labels
    sample_rate = int(df.shape[0]/12)
    plt.xticks(df[date_column][::sample_rate], rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{symbol}: Close Price and SMA_20 Over Time")
    plt.legend()
    plt.grid(True)
        
    if outfile:
        plt.savefig(outfile, dpi=300)

    if display:
        plt.show()

    plt.close()



from scipy.optimize import curve_fit

# Define model functions

def exponential(x, a, b, c):
    """Exponential function: a * exp(b*x) + c"""
    return a * np.exp(b * x) + c

def power_law(x, a, b, c):
    """Power law function: a * x^b + c"""
    return a * np.power(x, b) + c

def poly1(x, a, b):
    """1st order polynomial: a*x + b"""
    return a * x + b

def poly2(x, a, b, c):
    """2nd order polynomial: a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def poly3(x, a, b, c, d):
    """3rd order polynomial: a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def poly4(x, a, b, c, d, e):
    """4th order polynomial: a*x^4 + b*x^3 + c*x^2 + d*x + e"""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def poly5(x, a, b, c, d, e, f):
    """5th order polynomial: a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f"""
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

def poly6(x, a, b, c, d, e, f, g):
    """6th order polynomial: a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g"""
    return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g


# Logistic function
def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def fit_model(model, xdata, ydata, p0=None):
    """
    Fits a model function to data using scipy.optimize.curve_fit.

    p0: array-like, optional
        Initial guess for the parameters.
            
    Returns:
        popt: array
            Optimized parameters.
        pcov: 2d array
            The estimated covariance of popt.
    """
    popt, pcov = curve_fit(model, xdata, ydata, p0=p0)
    return popt, pcov


def plot_best_fit_with_error(x, y, popt, pcov, model, model_name):
    # Compute the best-fit model values
    y_fit = model(x, *popt)
    
    # Compute the Jacobian using finite differences
    n_params = len(popt)
    J = np.zeros((len(x), n_params))
    delta = 1e-8  # a small step for numerical derivative
    for i in range(n_params):
        popt_step = np.copy(popt)
        popt_step[i] += delta
        y_step = model(x, *popt_step)
        J[:, i] = (y_step - y_fit) / delta
    
    # Propagate the uncertainties to get variance at each x:
    # variance = diag(J @ pcov @ J.T)
    y_var = np.sum((J @ pcov) * J, axis=1)
    y_sigma = np.sqrt(y_var)
    
    # Plot the data, the best-fit line, and the uncertainty region
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Stock price", color="royalblue")
    plt.plot(x, y_fit, label=model_name, color="red")
    plt.fill_between(x, y_fit - y_sigma, y_fit + y_sigma, color="red", alpha=0.3, label="Fit uncertainty")
    plt.xlabel("Time [arb units]")
    plt.ylabel("Stock Price [$]")
    plt.title(f"Best fit to stock using {model_name}")
    plt.legend()
    plt.ylim(np.min(y)*0.9, np.max(y)*1.0)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_fft_components(X_filtered, N, title_prefix=""):
    """
    Plot the individual sine/cosine components kept in a filtered FFT.
    
    Parameters:
    - X_filtered: Filtered FFT (complex-valued, length N)
    - N: Number of time samples (length of original signal)
    """
    t = np.arange(N)
    freqs = np.fft.fftfreq(N)

    # Reconstruct full signal from filtered FFT
    full_signal = np.fft.ifft(X_filtered).real

    # Prepare figure
    plt.figure(figsize=(12, 8))
    plt.plot(t, full_signal, label='Reconstructed Signal', linewidth=2, color='black', alpha=0.8)

    # Plot each retained component - only for positive frequencies
    for k in range(N // 2 + 1):  # only plot 0 to N/2
        Xk = X_filtered[k]
        if np.abs(Xk) > 1e-8:
            component = np.exp(1j * 2 * np.pi * k * t / N) * Xk / N
            plt.plot(t, component.real, linestyle='--', label = f'k={k} | f={freqs[k]:.5f} 1/s | T={1/freqs[k]:.2f} days' if freqs[k] != 0 else f'k={k} | f=0 (DC)')

    plt.title(f"{title_prefix} Individual Components of Filtered FFT")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np

def fourier_forecast(X_trunc, H):
    """
    Given a truncated complex spectrum X_trunc of length N,
    return H future forecasted points (real-valued).
    """
    N = len(X_trunc)
    # time indices for future steps: N, N+1, ..., N+H-1
    n = np.arange(N, N+H)
    # frequency bin indices: 0,1,...,N-1
    k = np.arange(N)
    # Outer product nk gives matrix of size H×N
    # We compute e^{j2π k n / N} for all (n,k)
    exponent = np.exp(1j * 2 * np.pi * np.outer(n, k) / N)
    # Multiply each column k by X_trunc[k], sum over k, divide by N
    x_future = (exponent * X_trunc).sum(axis=1) / N
    return x_future.real
