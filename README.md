# Handling-Dirty-Data-Methodology-

## High-Frequency Data Cleaning (Python)

This repo contains a single-file, performance-oriented demo of industrial-grade market data cleaning.

Files:
- `clean_market_data_accelerated.py`: vectorized cleaning pipeline + benchmark (accelerated vs. Python-loop baseline)

## Handling Dirty Data (Methodology)

Below is the methodology used to handle high-frequency "dirty ticks" and microstructure noise.

### Method 1: Outlier Detection via Rolling Z-Score

Use a rolling window to compute robust z-scores on returns (or price changes) and flag extreme deviations.
In practice, prefer robust statistics (median/MAD) over mean/std, and always gate by market regime
(volatility and spread) to avoid deleting true price discovery.

### Method 2: Bounce-back signal filtering

**Bounce-Back Filter.** A classic bad tick is a single print that spikes away from the local level and then
immediately reverts on the next tick. We detect the 3-point pattern `p[t-1] -> p[t] -> p[t+1]` where the
initial jump is large, the next move flips sign, and `p[t+1]` returns close to `p[t-1]`. This removes
isolated feed glitches while preserving genuine level shifts and trending moves.

**Spread Validation.** For each trade, validate that the print price lies inside the contemporaneous best
bid/ask interval (allowing a small tolerance `eps` if needed). Prints outside the spread are often caused by
stale quotes, feed desynchronization, or misreported trade conditions, and should be excluded from feature
generation and labeling.
