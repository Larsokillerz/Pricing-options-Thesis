import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def CRR_model(S0, K, T, N, r, sig, option='eu', kind='P'):
    """Calculate the option prices using the CRR model."""
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialise all stock prices in the tree.
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    # Initialise the payoff
    if kind == 'P':
        payoff = np.maximum(K - S, np.zeros(N + 1))
    if kind == 'C':
        payoff = np.maximum(S - K, np.zeros(N + 1))

    # Recursion backwards through the tree.
    if option == 'eu':
        for i in np.arange(N, 0, -1):
            payoff = np.exp(-r * dt) * (p * payoff[1:i+1] + (1 - p) * payoff[0:i])
    elif option == 'am':
        for i in np.arange(N-1, -1, -1):
            S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            payoff[:i + 1] = np.exp(-r * dt) * (p * payoff[1:i+2] + (1 - p) * payoff[0:i+1])
            payoff = payoff[:-1]
            if kind == 'P':
                payoff = np.maximum(payoff, K - S)
            if kind == 'C':
                payoff = np.maximum(payoff, S - K)

    return payoff[0]

def blackScholes_model(S, K, T, r, sig, option='P'):
    """Calculate the option price using the Black-Scholes formula."""
    d1 = (np.log(S/K) + (r + ((sig**2)/2))*T)/(sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if option == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    if option == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def plot_price_eu(S0, K, T, r, sig):
    """Plot the price of the European options"""
    N = np.arange(5, 200, 1)
    price = [CRR_model(S0, K, T, i, r, sig) for i in N]
    bs = [blackScholes_model(S0, K, T, r, sig) for _ in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, bs, label='BS model')
    plt.plot(N, price, label='Binomial model')
    plt.xlabel('N')
    plt.ylabel('Price')
    plt.title('European option price.')
    plt.legend()
    plt.show()

def plot_am(S0, K, T, r, sig):
    """Plot the price of the American options."""
    N = np.arange(5, 200, 1)
    price = [CRR_model(S0, K, T, i, r, sig, 'am') for i in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, price, label='Binomial model')
    plt.xlabel('N')
    plt.ylabel('Price')
    plt.title('American option price.')
    plt.legend()
    plt.show()

def error_eu(S0, K, T, r, sig):
    """Plot the error of the European options."""
    price = blackScholes_model(S0, K, T, r, sig)
    h = np.arange(10, 1000, 1)
    error = []
    for i in h:
        new_price = CRR_model(S0, K, T, i, r, sig)
        error.append(np.linalg.norm(new_price - price))
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(h, (2.5/h), label="1/x convergence")
    plt.plot(h, error, label="error")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Error against BS.')
    plt.legend()
    plt.show()

def error_am(S0, K, T, r, sig):
    """Plot the error of the American options."""
    price = CRR_model(S0, K, T, 10000, r, sig, 'am')
    h = np.arange(10, 1000, 1)
    error = []
    for i in h:
        new_price = CRR_model(S0, K, T, i, r, sig, 'am')
        error.append(np.linalg.norm(new_price - price))
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(h, 2/h, label="1/x convergence")
    plt.plot(h, error, label="error")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Error American option.')
    plt.legend()
    plt.show()

def plot_am_eu(S0, K, T, r, sig):
    """Plot the price of the European options and
    American options in one figure.
    """
    N = np.arange(5, 200, 1)
    price_eu = [CRR_model(S0, K, T, i, r, sig, 'eu', 'P') for i in N]
    price_am = [CRR_model(S0, K, T, i, r, sig, 'am', 'P') for i in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, price_eu, label='European price')
    plt.plot(N, price_am, label='American price')
    plt.xlabel('N')
    plt.ylabel('Price')
    plt.title('European and American put option price.')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Initialize all begin values.
    S0 = 100
    K = 105
    T = 1
    N = 100
    r = 0.05
    sig = 0.2
    # Uncomment one of the following lines:
    # plot_price_eu(S0, K, T, r, sig)
    # error_eu(S0, K, T, r, sig)
    # plot_am(S0, K, T, r, sig)
    # error_am(S0, K, T, r, sig)
    # plot_am_eu(S0, K, T, r, sig)