from pricing import black_scholes_price
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def main():
    # ----------------------------------------------
    # Definiing the Underlying Asset
    # ----------------------------------------------
    stock = yf.Ticker("AAPL")
    price = stock.fast_info['lastPrice']

    # ----------------------------------------------
    # Calculating Volatility using 3 months of historical data
    # ----------------------------------------------
    hist = stock.history(period = "3mo")

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1))
    log_returns = log_returns.dropna()

    sigma = log_returns.std() * np.sqrt(252) # Annualized volatility

    # ----------------------------------------------
    # Option Parameters
    # ----------------------------------------------
    K = 275 # Using an ATM strike price for testing
    expiry = datetime(2026, 12, 18)

    today = datetime.today()
    T = (expiry - today).days / 365.0 # Time to maturity in years

    if T <= 0:
        raise ValueError("Expiry date must be in the future")

    t_bill = yf.Ticker("^IRX") # Using 13-week T-bill rate as risk-free rate proxy
    r = t_bill.fast_info["lastPrice"] / 100.0 # Convert percentage to decimal
    
    # ----------------------------------------------
    # Calculating Option Price
    # ----------------------------------------------
    call_price = black_scholes_price(price, K, T, r, sigma, option_type="call")
    put_price = black_scholes_price(price, K, T, r, sigma, option_type="put")

    # ----------------------------------------------
    # Output Results
    # ----------------------------------------------
    print(f"Underlying (S): {price:.2f}")
    print(f"Strike (K): {K}")
    print(f"Time to maturity (T): {T:.3f} years")
    print(f"Volatility (σ): {sigma:.2%}")
    print(f"Risk-free rate (r): {r:.2%}")
    print()
    print(f"Call price: {call_price:.2f}")
    print(f"Put price:  {put_price:.2f}")

if __name__ == "__main__":
    main()