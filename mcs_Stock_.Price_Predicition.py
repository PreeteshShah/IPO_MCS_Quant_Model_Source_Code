import numpy as np
import pandas as pd

# Constants and assumptions
initial_ebitda = 290.759 * 1_000_000  # Initial EBITDA in dollars
net_debt = 666_897_000  # Current Net Debt in dollars
shares_outstanding = 65.85 * 1_000_000  # Shares outstanding
simulations = 100_000  # Number of Monte Carlo simulations

# Scenario-specific EV/EBITDA multiples (mean ± std deviation for variability)
multiples = {
    "Base Case": {"mean": 12.00, "std_dev": 1.0},
    "Upside Case": {"mean": 15.00, "std_dev": 1.0},
    "Downside Case": {"mean": 8.50, "std_dev": 0.5}
}

# EBITDA growth assumptions (annual growth rate: mean ± std deviation)
ebitda_growth = {"mean": 0.05, "std_dev": 0.02}  # 5% mean growth with 2% variability

# Time horizon: 2 years
time_horizon = 2

# Results dictionary to store stock price simulations for each case
stock_price_results = {}

# Monte Carlo Simulation for each scenario
for scenario, params in multiples.items():
    # Simulate EBITDA growth rates for 2 years
    growth_rates = np.random.normal(ebitda_growth["mean"], ebitda_growth["std_dev"], (simulations, time_horizon))
    ebitda_simulated = initial_ebitda * np.prod(1 + growth_rates, axis=1)  # Simulated EBITDA after 2 years
    
    # Simulate EV/EBITDA multiples
    ev_ebitda_simulated = np.random.normal(params["mean"], params["std_dev"], simulations)
    
    # Calculate Enterprise Value (EV) for each simulation
    ev_simulated = ebitda_simulated * ev_ebitda_simulated
    
    # Calculate Market Capitalization (Market Cap)
    market_cap_simulated = ev_simulated - net_debt
    
    # Calculate Stock Price
    stock_price_simulated = market_cap_simulated / shares_outstanding
    
    # Store results for this scenario
    stock_price_results[scenario] = stock_price_simulated

# Convert results to a DataFrame for analysis
stock_price_df = pd.DataFrame(stock_price_results)

# Summarize results: mean, 5th percentile, 95th percentile
summary_stats = stock_price_df.describe(percentiles=[0.05, 0.95]).T[["mean", "5%", "95%"]]
summary_stats.columns = ["Mean Stock Price", "5th Percentile", "95th Percentile"]

# Display results
# Display the summary statistics in the console
print("Monte Carlo Stock Price Predictions for Scenarios:")
print(summary_stats)

# Save the summary to a CSV file
summary_stats.to_csv("monte_carlo_stock_price_predictions.csv", index=True)
print("Results saved to 'monte_carlo_stock_price_predictions.csv'")


import matplotlib.pyplot as plt

# Plot all simulated stock prices for each scenario
plt.figure(figsize=(12, 8))

# Randomly sample 500 simulations from the 10,000 for clarity in plotting
sampled_data = stock_price_df.sample(500, axis=0)

for scenario in sampled_data.columns:
    plt.plot(range(len(sampled_data[scenario])), sampled_data[scenario], alpha=0.5, label=scenario)

# Add labels, legend, and title
plt.title("Monte Carlo Simulations of Stock Prices for Scenarios", fontsize=16)
plt.xlabel("Simulation Number (Sampled)", fontsize=14)
plt.ylabel("Stock Price ($)", fontsize=14)
plt.legend(title="Scenarios", fontsize=12)
plt.grid(True)
plt.show()


