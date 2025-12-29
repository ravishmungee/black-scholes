import math
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """
    Internal helper to compute d1 and d2
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def delta(S, K, T, r, sigma, option_type="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)

    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Returns Vega per 1.00 change in volatility (not 1%)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)

    return S * norm.pdf(d1) * math.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Returns Theta per year
    (divide by 365 for per-day theta)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    first_term = - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))

    if option_type == "call":
        second_term = - r * K * math.exp(-r * T) * norm.cdf(d2)
        return first_term + second_term

    elif option_type == "put":
        second_term = r * K * math.exp(-r * T) * norm.cdf(-d2)
        return first_term + second_term

    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, option_type="call"):
    """
    Returns Rho per 1.00 change in interest rate (not 1%)
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")