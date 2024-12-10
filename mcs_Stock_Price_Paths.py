# Reimport necessary libraries due to reset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Reinitialize required data and parameters
current_stock_price = 37.19  # Current stock price
new_drift = 0.2  # Increased drift (20% growth rate)
new_volatility = 0.15  # Reduced volatility (15% annualized)
time_horizon = 2  # Time horizon in years
num_simulations = 100_000  # Number of simulations
time_steps = int(252 * time_horizon)  # Number of trading days for 2 years
dt = 1 / 252  # Time step (1 trading day)

# Target prices for scenarios
price_targets = {
    "Base Case": 48.38,
    "Upside Case": 62.88,
    "Downside Case": 31.29
}

# Simulate stock price paths with adjusted parameters
new_simulated_prices = np.zeros((num_simulations, time_steps))
new_simulated_prices[:, 0] = current_stock_price

for t in range(1, time_steps):
    random_shocks = np.random.normal(0, 1, num_simulations)
    new_simulated_prices[:, t] = new_simulated_prices[:, t - 1] * np.exp(
        (new_drift - 0.5 * new_volatility**2) * dt
        + new_volatility * np.sqrt(dt) * random_shocks
    )

# Prepare sample simulations for visualization (e.g., 500 paths)
sample_size = 500
sample_indices = np.random.choice(range(num_simulations), sample_size, replace=False)
sampled_prices = new_simulated_prices[sample_indices, :]

# Plot stock price paths with green lines
plt.figure(figsize=(12, 8))

# Plot individual paths with green lines
for i in range(sample_size):
    plt.plot(np.linspace(0, time_horizon, time_steps), sampled_prices[i], alpha=0.5, lw=0.5, color="green")

# Highlight the mean stock price path
mean_path = new_simulated_prices.mean(axis=0)
plt.plot(np.linspace(0, time_horizon, time_steps), mean_path, color="red", lw=2, label="Mean Path")

# Add horizontal lines for target prices
for scenario, target in price_targets.items():
    plt.axhline(y=target, linestyle="--", label=f"{scenario} Target: ${target:.2f}", color="orange")

# Add labels, legend, and title
plt.title("Monte Carlo Simulated Stock Price Paths Over 2 Years (Green Paths)", fontsize=16)
plt.xlabel("Time (Years)", fontsize=14)
plt.ylabel("Stock Price ($)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Calculate percentages of simulated paths in each range

# Final prices from simulations (end of 2 years)
final_prices = new_simulated_prices[:, -1]

# Calculate percentages for each range
percent_above_upside = np.mean(final_prices >= price_targets["Upside Case"]) * 100
percent_between_upside_base = np.mean(
    (final_prices >= price_targets["Base Case"]) & (final_prices < price_targets["Upside Case"])
) * 100
percent_between_base_downside = np.mean(
    (final_prices >= price_targets["Downside Case"]) & (final_prices < price_targets["Base Case"])
) * 100
percent_below_downside = np.mean(final_prices < price_targets["Downside Case"]) * 100

# Create a summary of the results
range_summary = pd.DataFrame({
    "Range": [
        "Above Upside Case",
        "Between Upside and Base Case",
        "Between Base and Downside Case",
        "Below Downside Case"
    ],
    "Percentage (%)": [
        percent_above_upside,
        percent_between_upside_base,
        percent_between_base_downside,
        percent_below_downside
    ]
})

# Display the results
print("Must close graph to see stats")
print(range_summary)
range_summary.to_csv("percentage_simulations_by_price_ranges.csv", index=True)
print("Results saved to 'percentage_simulations_by_price_ranges.csv'")
