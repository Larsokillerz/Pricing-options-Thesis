import numpy as np
import matplotlib.pyplot as plt
import time

def CRR_model_complexity_analysis(S0, K, T, n, r, sig, option='eu'):
    N = ((n + 1) * (n + 2)) // 2
    time_arr_eu = []
    time_arr_am = []
    for i in n:
        dt = T/i
        u = np.exp(sig * np.sqrt(dt))
        d = 1/u
        p = (np.exp(r * dt) - d) / (u - d)

        # Initialise all stock prices in the tree.
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))

        # Initialise the payoff
        payoff = np.maximum(K - S, np.zeros(i + 1))

        # Recursion backwards through the tree.
        if option == 'eu':
            time_start = time.time()
            for j in np.arange(i, 0, -1):
                payoff = np.exp(-r * dt) * (p * payoff[1:j+1] + (1 - p) * payoff[0:j])
            option_prices = [calculate_option_price(node) for node in S]
            time_elapsed = time.time() - time_start
            time_arr_eu.append(time_elapsed)
        elif option == 'am':
            time_start = time.time()
            for l in np.arange(i-1, -1, -1):
                S = S0 * d ** (np.arange(l, -1, -1)) * u ** (np.arange(0, l + 1, 1))
                payoff[:l + 1] = np.exp(-r * dt) * (p * payoff[1:l+2] + (1 - p) * payoff[0:l+1])
                payoff = payoff[:-1]
                payoff = np.maximum(payoff, K - S)
            option_prices = [calculate_option_price(node) for node in S]
            time_elapsed = time.time() - time_start
            time_arr_am.append(time_elapsed)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, time_arr_eu, label="time")
    plt.xlabel('Number of nodes (N)')
    plt.ylabel('Computational time (seconds)')
    plt.title('Time Complexity of pricing option in binomial model')
    plt.grid(True)
    plt.legend()
    plt.show()

def time_analysis_construction_tree(S0, n, T, beta):
    time_arr = []
    N = ((n + 1) * (n + 2)) // 2
    for i in n :
        dt = T/i
        u = np.exp(beta * np.sqrt(dt))
        d = 1/u
        time_start = time.time()
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        option_prices = [calculate_option_price(node) for node in S]
        time_elapsed = time.time() - time_start
        time_arr.append(time_elapsed)
    plt.figure(figsize=(7,7), dpi=250)
    plt.plot(N, time_arr, label="time")
    plt.xlabel('Number of nodes (N)')
    plt.ylabel('Computational time (seconds)')

    plt.title('Time Complexity of construction binomial tree')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_option_price(node):
    """Simulating some computational time.
    """
    time.sleep(0.001)
    return node


if __name__ == '__main__':
    beta = 0.15
    S0 = 100
    K = 105
    T = 1
    N = 1000
    r = 0.05
    sig = 0.15
    n = np.arange(1, 200, 1)
    time_analysis_construction_tree(S0, n, T, beta)
    # CRR_model_complexity_analysis(S0, K, T, n, r, sig, option='eu')
