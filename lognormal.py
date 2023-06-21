import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scp
import scipy as sp
import seaborn as sns
from greek_letter import flatten_list

def lognormal_GBM(S0, alpha, beta, T, h):
    """Calculate the lognormal pdf with the right paramters.
    """
    sig = np.sqrt((beta**2) * T)
    mu = np.log(S0) + (alpha - ((beta**2)/2)) * T
    S = (np.exp(-((np.log(h) - mu)**2 / (2 * (sig**2)))) / (h * sig * np.sqrt(2 * np.pi)))
    return S

def binom_prob(k, n, p):
    """Calculate the binomial pdf.
    """
    coef = sp.special.binom(n, k)
    binom = coef * (p**k) * ((1 - p)**(n - k))
    return binom

def BT_approx_GBM(alpha, beta, T, k, n, S0):
    """Calculate the binomial probability with the right parameters and
    calculate the value of S.
    """
    dt = T/n
    u = np.exp(beta * np.sqrt(dt))
    d = 1/u
    p = (1/2) + (alpha/(2*beta)) * np.sqrt(dt)
    # p = (np.exp(alpha * dt) - d) / (u - d)
    S = S0 * d ** (np.arange(n, -1, -1)) * u ** (np.arange(0, n + 1, 1))
    binom = binom_prob(k, n, p)
    return S, binom

def BT_approx_GBM_extend(alpha, beta, T, k, n, S0):
    """Calculate the binomial probability with the right parameters and
    calculate the value of S.
    """
    dt = T/n
    u = np.exp(beta * np.sqrt(dt))
    d = 1/u
    # p = (1/2) + (alpha/(2*beta)) * np.sqrt(dt)
    p = (np.exp(alpha * dt) - d) / (u - d)
    S = S0 * d ** (np.arange(n, -1, -1)) * u ** (np.arange(0, n + 1, 1))
    binom = binom_prob(k, n, p)
    return S, binom

def plot(S, BT, h):
    """Plot lognormal probability against binomial probability.
    """
    fig, ax1 = plt.subplots(figsize=(7, 7), dpi = 200)
    ax1.set_xlabel('Value of S')
    ax1.set_ylabel('Lognormal probability')
    ax1.plot(h, S, label="Lognormal GBM", color='red')

    ax2 = ax1.twinx()
    k, B = BT
    ax2.set_ylabel('Binomial probability')
    ax2.plot(k, B, 's', label="Binomial", color='black')
    fig.suptitle('Binomial tree against lognormal GBM')
    fig.legend()
    plt.grid(axis = 'y')
    plt.show()

def vary(S0, T, per, h, k):
    """Vary alpha and beta and plot it into subplots.
    """
    alpha = np.linspace(0.1, 0.3, num=4)
    # alpha = 0.15
    beta = np.linspace(0.1, 0.3, num=4)
    # beta = 0.15
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    for i, j in zip(alpha, axs.flatten()):
        BT = BT_approx_GBM(i, beta, T, k, per, S0)
        lognorm = lognormal_GBM(S0, i, beta, T, h)
        plot1 = j.plot(h, lognorm, label="Lognormal GBM", color='red')

        ax2 = j.twinx()
        new_k, B = BT

        plot2= ax2.plot(new_k, B, 's', label="Binomial", color='black')
        j.set_title(f"Alpha = {i}")
        plots = plot1 + plot2
        labels = [k.get_label() for k in plots]
        j.grid(axis = 'y')

    fig.tight_layout(pad=5.0)
    fig.supxlabel('Value of S')
    fig.supylabel('Lognormal probability')
    fig.text(x=0.97, y=0.5, s="Binomial probability", size=12, rotation=270, ha='center', va='center')
    fig.suptitle('Binomial tree against lognormal GBM', y=1)
    fig.legend(plots, labels, loc='upper left')
    plt.show()


def plot_even_odd(S, h, alpha, beta, T, S0):
    """Plot lognormal probability against binomial probability.
    """
    fig, ax1 = plt.subplots(figsize=(7, 7), dpi = 200)
    ax1.set_xlabel('Value of S')
    ax1.set_ylabel('Lognormal probability')
    ax1.plot(h, S, label="Lognormal GBM", color='red')

    per = np.arange(100, 101 + 1, 1)
    s_even, s_odd, B_even, B_odd = [], [], [], []
    for i in per:
        k = np.arange(0, i + 1, 1)
        if i % 2 == 0:
            S, B = BT_approx_GBM(alpha, beta, T, k, i, S0)
            s_even.append(S.tolist())
            B_even.append(B.tolist())
        else:
            S, B = BT_approx_GBM(alpha, beta, T, k, i, S0)
            s_odd.append(S.tolist())
            B_odd.append(B.tolist())
    s_even = flatten_list(s_even)
    s_odd = flatten_list(s_odd)
    B_even = flatten_list(B_even)
    B_odd = flatten_list(B_odd)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Binomial probability')
    ax2.plot(s_odd, B_odd,label="Binomial odd")
    ax2.plot(s_even, B_even, label="Binomial even")
    fig.suptitle('Binomial tree against lognormal GBM')
    fig.legend()
    plt.grid(axis = 'y')
    plt.show()

if __name__ == '__main__':
    # Initialize the begin values.
    S0 = 100
    T = 1
    N = 350
    per = 31
    k = np.arange(0, per + 1, 1)
    h = np.arange(1, N, 1)
    alpha = 0.15
    beta = 0.15
    lognorm = lognormal_GBM(S0, alpha, beta, T, h)
    BT = BT_approx_GBM(alpha, beta, T, k, per, S0)
    # plot(lognorm, BT, h)
    # vary(S0, T, per, h, k)
    plot_even_odd(lognorm, h, alpha, beta, T, S0)