import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from pricing import blackScholes_model, CRR_model

def flatten_list(lst):
    flattened = []
    for i in lst:
        if isinstance(i, list):
            flattened.extend(flatten_list(i))
        else:
            flattened.append(i)
    return flattened

def delta(S0, K, T, r, N, sig, option='eu', kind='P'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)
    # Initialise all stock prices in the tree.
    S = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))

    Su = S0 * u
    Sd = S0 * d

    # Initialise the payoff
    if kind == 'P':
        payoff = np.maximum(K - S, np.zeros(N + 1))
    if kind == 'C':
        payoff = np.maximum(S - K, np.zeros(N + 1))

    if option == 'eu':
        for i in np.arange(N, 1, -1):
            payoff = np.exp(-r * dt) * (p * payoff[1:i+1] + (1 - p) * payoff[0:i])
    if option == 'am':
        for i in np.arange(N-1, 0, -1):
            S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            payoff[:i + 1] = np.exp(-r * dt) * (p * payoff[1:i+2] + (1 - p) * payoff[0:i+1])
            payoff = payoff[:-1]
            if kind == 'P':
                payoff = np.maximum(payoff, K - S)
            if kind == 'C':
                payoff = np.maximum(payoff, S - K)
    Delta = (payoff[1] - payoff[0]) / (Su - Sd)
    return Delta

def plot_delta_both(S0, K, T, r, sig, kind):
    N = np.arange(5, 100, 1)
    d_am = [delta(S0, K, T, r, i, sig, 'am', kind) for i in N]
    d_eu = [delta(S0, K, T, r, i, sig, 'eu', kind) for i in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, d_am, label='Delta American option')
    plt.plot(N, d_eu, label='Delta European option')
    plt.xlabel('N')
    plt.ylabel('Delta')
    plt.title('Delta for American call and European call options')
    plt.legend()
    plt.show()

def plot_delta(S0, K, T, r, sig, option, kind):
    N = np.arange(5, 100, 1)
    d1 = (np.log(S0/K) + (r + ((sig**2)/2))*T)/(sig * np.sqrt(T))
    if kind == 'C':
        delta_ana = norm.cdf(d1)
    if kind == 'P':
        delta_ana = norm.cdf(d1) - 1
    bs = [delta_ana for _ in N]
    d = [delta(S0, K, T, r, i, sig, option, kind) for i in N]
    x_even, x_odd, new_d1_even, new_d1_odd = [], [], [], []
    for i in N:
        if i % 2 == 0:
            new_d1_even.append(delta_analytical_eu(S0, K, T, sig, r, i, kind))
            x_even.append(i)
        else:
            new_d1_odd.append(delta_analytical_eu(S0, K, T, sig, r, i, kind))
            x_odd.append(i)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, d, label='Delta')
    plt.plot(x_even, new_d1_even, label='even')
    plt.plot(x_odd, new_d1_odd, label='odd')
    plt.xlabel('N')
    plt.ylabel('Delta')
    if option == 'eu' and kind == 'C':
        plt.title('Delta for different values of N for European call option')
    if option == 'am' and kind == 'C':
        plt.title('Delta for different values of N for American call option')
    if option == 'eu' and kind == 'P':
        plt.title('Delta for different values of N for European put option')
    if option == 'am' and kind == 'P':
        plt.title('Delta for different values of N for American put option')
    plt.legend()
    plt.show()


def error_analytical(S0, K, T, sig, r, N):
    dt = T/N
    T = T - dt
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    a = ((np.log(K / S0)) - (N * np.log(d))) / (np.log(u) - np.log(d))
    frac = a % 1
    first_part_D_1 = (1/(96 * sig * np.sqrt(T)))
    part_2_D_1 = 4 * (np.log(S0 / K))**2
    part_3_D_1 = 8 * r * T * np.log(S0 / K)
    part_4_D_1 = 3 * T * ((4 * (sig**2)) - (12 * r * T**2) - ((sig**4) * T))
    D_1 = first_part_D_1 * (part_2_D_1 - part_3_D_1 + part_4_D_1)
    eq = sig * np.sqrt(T) * frac * (frac - 1)
    new_eq = eq + D_1
    d_1 = (1/(sig * np.sqrt(T))) * (np.log(S0 / K) + (r + ((sig**2) / 2)) * T)
    P_n = S0 * np.exp(-(((d_1)**2) / 2)) * np.sqrt(2/(np.pi)) * (new_eq) * (1/N)
    return P_n

def delta_analytical_eu(S0, K, T, sig, r, N, kind='C'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    price_up = blackScholes_model(S0 * u, K, T - dt, r, sig, kind) - error_analytical(S0 * u, K, T - dt, sig, r, N-1)
    price_down = blackScholes_model(S0 * d, K, T - dt, r, sig, kind) - error_analytical(S0 * d, K, T - dt, sig, r, N-1)
    Su = S0 * u
    Sd = S0 * d
    delta_ana = (price_up - price_down) / (Su - Sd)
    return delta_ana


if __name__ == '__main__':
    S0 = 100
    K = 105
    T = 1
    N = 100
    r = 0.05
    sig = 0.2
    # plot_delta(S0, K, T, sig, option='am', kind='P')
    plot_delta_both(S0, K, T, r, sig, kind='C')
    # delta(S0, K, T, N, sig, kind='P')
    # delta_analytical(S0, K, T, sig, r, N)