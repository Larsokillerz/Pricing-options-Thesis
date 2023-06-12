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
    plt.title('Delta for American put and European put options')
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

def plot_delta_analytical_am(S0, K, T, r, sig, kind):
    N = np.arange(5, 100, 1)
    delta_binomial = [delta(S0, K, T, r, i, sig, 'am', kind) for i in N]
    x_even, x_odd, new_d1_even, new_d1_odd = [], [], [], []
    for i in N:
        dt = T/i
        u = np.exp(sig * np.sqrt(dt))
        d = 1/u
        right_price_up = CRR_model(S0 * u, K, T - dt, 2000 - 1, r, sig, 'am', kind)
        right_price_down = CRR_model(S0 * d, K, T - dt, 2000 - 1, r, sig, 'am', kind)
        price_up = right_price_up - error_analytical(S0 * u, K, T - dt, sig, r, i-1)
        price_down = right_price_down - error_analytical(S0 * d, K, T - dt, sig, r, i-1)
        Su = S0 * u
        Sd = S0 * d
        delta_ana = (price_up - price_down) / (Su - Sd)
        if i % 2 == 0:
            new_d1_even.append(delta_ana)
            x_even.append(i)
        else:
            new_d1_odd.append(delta_ana)
            x_odd.append(i)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, delta_binomial, label='Delta')
    plt.plot(x_even, new_d1_even, label='even')
    plt.plot(x_odd, new_d1_odd, label='odd')
    plt.xlabel('N')
    plt.ylabel('Delta')
    if kind == 'C':
        plt.title('Delta for different values of N for American call option')
    if kind == 'P':
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

def gamma(S0, K, T, r, N, sig, option='eu', kind='P'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    delta_up = delta(S0 * u, K, T, r, N - 1, sig, option, kind)
    delta_down = delta(S0 * d, K, T, r, N - 1, sig, option, kind)
    Su = S0 * (u**2)
    Sd = S0 * (d**2)
    return (delta_up - delta_down) / ((Su - Sd) / 2)

def gamma_exact(S0, K, T, r, sig):
    d1 = (np.log(S0/K) + (r + ((sig**2)/2))*T)/(sig * np.sqrt(T))
    N_acc = (1 / (np.sqrt(2 * np.pi))) * np.exp(-(d1**2 / 2))
    G = (N_acc / (S0 * sig * np.sqrt(T)))
    return G

def plot_gamma(S0, K, T, r, sig, option, kind):
    N = np.arange(5, 200, 1)
    gamma_binomial = [gamma(S0, K, T, r, i, sig, option, kind) for i in N]
    gamma_ana = [gamma_exact(S0, K, T, r, sig) for _ in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, gamma_binomial, label='Gamma binomial tree')
    plt.plot(N, gamma_ana, label='Gamma exact')
    plt.xlabel('N')
    plt.ylabel('Gamma')
    plt.title('Gamma for European call option')
    plt.legend()
    plt.show()

def gamma_analytical(S0, K, T, sig, r, N, kind='C'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    Su = S0 * (u**2)
    Sd = S0 * (d**2)
    delta_up = delta_analytical_eu(S0 * u, K, T, sig, r, N - 1, kind)
    delta_down = delta_analytical_eu(S0 * d, K, T, sig, r, N - 1, kind)
    return (delta_up - delta_down) / ((Su - Sd) / 2)

def plot_gamma_analytical(S0, K, T, r, sig, option, kind):
    N = np.arange(5, 100, 1)
    gamma_binomial = [gamma(S0, K, T, r, i, sig, option, kind) for i in N]
    x_even, x_odd, new_d1_even, new_d1_odd = [], [], [], []
    for i in N:
        if i % 2 == 0:
            new_d1_even.append(gamma_analytical(S0, K, T, sig, r, i, kind))
            x_even.append(i)
        else:
            new_d1_odd.append(gamma_analytical(S0, K, T, sig, r, i, kind))
            x_odd.append(i)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, gamma_binomial, label='Gamma')
    plt.plot(x_even, new_d1_even, label='even')
    plt.plot(x_odd, new_d1_odd, label='odd')
    plt.xlabel('N')
    plt.ylabel('Gamma')
    if option == 'eu' and kind == 'C':
        plt.title('Gamma for different values of N for European call option')
    if option == 'am' and kind == 'C':
        plt.title('Gamma for different values of N for American call option')
    if option == 'eu' and kind == 'P':
        plt.title('Gamma for different values of N for European put option')
    if option == 'am' and kind == 'P':
        plt.title('Gamma for different values of N for American put option')
    plt.legend()
    plt.show()

def plot_gamma_both(S0, K, T, r, sig, kind):
    N = np.arange(5, 100, 1)
    gamma_am = [gamma(S0, K, T, r, i, sig, 'am', kind) for i in N]
    gamma_eu = [gamma(S0, K, T, r, i, sig, 'eu', kind) for i in N]
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, gamma_am, label='Gamma American option')
    plt.plot(N, gamma_eu, label='Gamma European option')
    plt.xlabel('N')
    plt.ylabel('Gamma')
    plt.title('Gamma for American put and European put options')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    S0 = 100
    K = 105
    T = 1
    N = 100
    r = 0.0
    sig = 0.2
    # plot_delta(S0, K, T, r, sig, option='eu', kind='P')
    # plot_delta_both(S0, K, T, r, sig, kind='P')
    # delta(S0, K, T, N, sig, kind='P')
    # delta_analytical_eu(S0, K, T, sig, r, N)
    # plot_gamma(S0, K, T, r, sig, option='eu', kind='C')
    plot_gamma_analytical(S0, K, T, r, sig, option='eu', kind='C')
    # plot_gamma_both(S0, K, T, r, sig, kind='P')
    # plot_delta_analytical_am(S0, K, T, r, sig, kind='P')