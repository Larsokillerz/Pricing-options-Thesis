import matplotlib.pyplot as plt
import numpy as np

def brown_motion(T, n, d, times):
    dt = T/n

    dB = np.random.normal(0, np.sqrt(dt), size=(n - 1, d))
    B0 = np.zeros(shape=(1, d))

    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    return B

def quadratic_var(B):
    return np.cumsum(np.power(np.diff(B, axis=0, prepend=0.), 2), axis=0)

def geometric_brownian_motion(T, n, d):
    mu = 0.1
    S0 = 0.001
    sigma = 0.3
    dt = T/n
    S_t = np.exp((mu - (sigma ** 2 / 2)) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=(n - 1, d)))
    S_t = np.vstack([np.ones(d), S_t])
    S_t = S0 * S_t.cumprod(axis=0)
    return S_t

if __name__ == '__main__':
    T = 1.0
    n = 10000
    d = 1
    times = np.linspace(0, T, n)
    B = brown_motion(T, n, d, times)
    GBM = geometric_brownian_motion(T, n, d)
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.plot(times, GBM)
    plt.show()
