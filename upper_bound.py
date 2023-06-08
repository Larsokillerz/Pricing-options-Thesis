import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pricing import CRR_model, blackScholes_model


def plot_analytical_eu(S0, K, T, r, sig, option='C'):
    N = np.arange(5, 100, 1)
    price = [CRR_model(S0, K, T, i, r, sig, 'eu', option) for i in N]
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    D_1_array, eq_array, eq2_array, P_n_odd_array = [], [], [], []
    P_n_even_array, x_even, x_odd = [], [], []
    for i in N:
        P_n, D_1, eq = error_analytical(S0, K, T, sig, r, i)
        if i % 2 == 0:
            if option == 'C':
                P_n_even_array.append(blackScholes_model(S0, K, T, r, sig, 'C') - P_n)
            if option == 'P':
                C = blackScholes_model(S0, K, T, r, sig, 'C') - P_n
                new_P_n = C + K * np.exp(-r * T) - S0
                P_n_even_array.append(new_P_n)
            x_even.append(i)
        else:
            if option == 'C':
                P_n_odd_array.append(blackScholes_model(S0, K, T, r, sig, 'C') - P_n)
            if option == 'P':
                C = blackScholes_model(S0, K, T, r, sig, 'C') - P_n
                new_P_n = C + K * np.exp(-r * T) - S0
                P_n_odd_array.append(new_P_n)
            x_odd.append(i)

        D_1_array.append(D_1)
        eq_array.append(eq * (1/i))
        eq2_array.append(eq)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(x_even, P_n_even_array, label='even')
    plt.plot(x_odd, P_n_odd_array, label='odd')

    # plt.plot(N, D_1_array, label='D_1')
    # plt.plot(N, eq2_array, label='eq')
    plt.plot(N, price, label='Binomial model')
    plt.xlabel('N')
    plt.ylabel('Price')
    plt.title('European option price.')
    plt.legend()
    plt.show()


def error_analytical(S0, K, T, sig, r, N):
    # mu = (1/2) * sig**2
    # u = 1 + (sig / (np.sqrt(N))) + (mu / N)
    # d = 1 - (sig / (np.sqrt(N))) + (mu / N)
    dt = T/N
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
    return P_n, D_1, eq

def upperbound(T, N, r, sig):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)
    a = np.exp(r * dt)
    b = np.exp((sig**2) * dt)
    mn_2 = p * u**2 + (1 - p) * d**2 - (np.exp(2 * (r - ((sig**2)/2)) * (T/N)))
    mn_3 = p * u**3 + (1 - p) * d**3 - (np.exp(3 * (r - ((sig**2)/2)) * (T/N)))
    pn = p * np.log(u) * (u - 1)**3 + (1 - p) * np.log(d) * (d - 1)**3
    upper = (N * (mn_2 + mn_3 + pn)) + (1/N)
    return mn_2, mn_3, pn, upper

def error_upperbound_eu(S0, K, T, r, sig):
    price = blackScholes_model(S0, K, T, r, sig)
    h = np.arange(10, 1000, 1)
    error = []
    upper_bound = []
    C = np.linspace(0, 2, num=100)
    best_C = []
    mn_2_arr = []
    mn_3_arr = []
    pn_arr = []
    for i in h:
        mn_2, mn_3, pn, upper = upperbound(T, i, r, sig)
        upper_bound.append(upper)
        # mn_2_arr.append(mn_2)
        # mn_3_arr.append(mn_3)
        # pn_arr.append(pn)
        new_price = CRR_model(S0, K, T, i, r, sig)
        error.append(np.linalg.norm(new_price - price))
    for j in C:
        if all(j * np.array(upper_bound) >= np.array(error)):
            best_C.append(j)
        else:
            continue
    C = min(best_C)
    print(C)
    error_upper = C * np.array(upper_bound)
    # print(mn_3_arr)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(h, error_upper, label=f"Error_upperbound with {C}")
    plt.plot(h, error, label="Error")
    # plt.plot(h, mn_2_arr, label="M_n^2")
    # plt.plot(h, mn_3_arr, label="M_n^3")
    # plt.plot(h, pn_arr, label="P_n")
    # plt.plot(h, 1/(h**2), label="1/n^2")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Error against BS.')
    plt.legend()
    plt.show()

def error_upperbound_am(S0, K, T, r, sig):
    price = CRR_model(S0, K, T, 10000, r, sig, 'am')
    h = np.arange(10, 1000, 1)
    error = []
    upper_bound = []
    C = np.linspace(0, 2, num=100)
    best_C = []
    mn_2_arr = []
    mn_3_arr = []
    pn_arr = []
    for i in h:
        mn_2, mn_3, pn, upper = upperbound(T, i, r, sig)
        upper_bound.append(upper)
        # mn_2_arr.append(mn_2)
        # mn_3_arr.append(mn_3)
        # pn_arr.append(pn)
        new_price = CRR_model(S0, K, T, i, r, sig, 'am')
        error.append(np.linalg.norm(new_price - price))
    for j in C:
        if all(j * np.array(upper_bound) >= np.array(error)):
            best_C.append(j)
        else:
            continue
    C = min(best_C)
    print(C)
    error_upper = C * np.array(upper_bound)
    # print(mn_3_arr)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(h, error_upper, label=f"Error_upperbound with {C}")
    plt.plot(h, error, label="Error")
    # plt.plot(h, mn_2_arr, label="M_n^2")
    # plt.plot(h, mn_3_arr, label="M_n^3")
    # plt.plot(h, pn_arr, label="P_n")
    # plt.plot(h, 1/(h**2), label="1/n^2")
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Error against BS.')
    plt.legend()
    plt.show()


def plot_analytical_am(S0, K, T, r, sig, option='C'):
    N = np.arange(5, 100, 1)
    price = [CRR_model(S0, K, T, i, r, sig, 'am', option) for i in N]
    right_price = CRR_model(S0, K, T, 10000, r, sig, 'am', option)
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = 1/u
    D_1_array, eq_array, eq2_array, P_n_odd_array = [], [], [], []
    P_n_even_array, x_even, x_odd = [], [], []
    for i in N:
        P_n, D_1, eq = error_analytical(S0, K, T, sig, r, i)
        if i % 2 == 0:
            P_n_even_array.append(right_price - P_n)
            x_even.append(i)
        else:
            P_n_odd_array.append(right_price - P_n)
            x_odd.append(i)

        D_1_array.append(D_1)
        eq_array.append(eq * (1/i))
        eq2_array.append(eq)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(x_even, P_n_even_array, label='even')
    plt.plot(x_odd, P_n_odd_array, label='odd')

    # plt.plot(N, D_1_array, label='D_1')
    # plt.plot(N, eq2_array, label='eq')
    plt.plot(N, price, label='Binomial model')
    plt.xlabel('N')
    plt.ylabel('Price')
    plt.title('American option price.')
    plt.legend()
    plt.show()

def plot_multiple_am(K, T, r, sig, option='C'):
    S0 = np.arange(0.8, 1.6, 0.1)
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 12), sharex='all', dpi=175)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle("American call option price", fontsize=18, y=0.95)

    for ax in axs:
        list_S0 = []
        for i, l in zip(S0, range(0, len(ax))):
            list_S0.append(i)
            N = np.arange(5, 100, 1)
            price = [CRR_model(i, K, T, j, r, sig, 'am', option) for j in N]
            right_price = CRR_model(i, K, T, 10000, r, sig, 'am', option)
            dt = T/N
            u = np.exp(sig * np.sqrt(dt))
            d = 1/u
            D_1_array, eq_array, eq2_array, P_n_odd_array = [], [], [], []
            P_n_even_array, x_even, x_odd = [], [], []
            for k in N:
                P_n, D_1, eq = error_analytical(i, K, T, sig, r, k)
                if k % 2 == 0:
                    P_n_even_array.append(right_price - P_n)
                    x_even.append(k)
                else:
                    P_n_odd_array.append(right_price - P_n)
                    x_odd.append(k)

                D_1_array.append(D_1)
                eq_array.append(eq * (1/i))
                eq2_array.append(eq)
            ax[l].plot(x_even, P_n_even_array, label='even')
            ax[l].plot(x_odd, P_n_odd_array, label='odd')
            ax[l].plot(N, price, label='Binomial model')
            ax[l].set_title(f'S0 = {i}')

        S0 = np.setdiff1d(S0, list_S0)

    lines = []
    labels = []
    for ax in fig.axes:
        Line, Label = ax.get_legend_handles_labels()
        lines.extend(Line)
        labels.extend(Label)

    fig.legend(lines[0:3], labels[0:3], loc='upper left')
    fig.supxlabel('N')
    fig.supylabel('Price')
    plt.show()

if __name__ == '__main__':
    S0 = 1
    K = 1.1
    T = 1
    N = 100
    r = 0.0
    sig = 0.2
    price = (CRR_model(S0, K, T, N, r, sig, 'eu'))
    # print(price)
    # print(blackScholes_model(S0, K, T, r, sig))
    # plot_analytical_eu(S0, K, T, r, sig, 'P')
    # error_upperbound_eu(S0, K, T, r, sig)
    # error_upperbound_am(S0, K, T, r, sig)
    # plot_analytical_eu(S0, K, T, r, sig, option='C')
    plot_multiple_am(K, T, r, sig, option='C')