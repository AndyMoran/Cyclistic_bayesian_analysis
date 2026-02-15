# Cyclistic Bayesian Pricing & Behaviour Simulation

This project uses Bayesian modelling to understand how Cyclistic riders behave and how pricing changes might affect future revenue. Instead of relying on averages or point estimates, the analysis models full probability distributions for trip duration, value‑per‑trip, and revenue. A Monte Carlo pricing simulation tests different elasticity assumptions and leads to a risk‑aware recommendation for a $0.16 per‑minute pilot price.

The aim is simple: build a clear, honest, uncertainty‑aware framework for pricing decisions.

## Project Highlights

**Bayesian Modelling of Rider Behaviour**

Trip duration and revenue follow right‑skewed patterns, so the model uses Log‑Normal likelihoods to capture long‑tail behaviour common in mobility data.

**Value‑Per‑Trip Economics**

Casual and Member riders generate revenue in different ways. The model simulates amortised membership fees across realistic usage patterns to make the comparison fair.

**Monte Carlo Pricing Simulation**

Future revenue is simulated under three elasticity assumptions — Low, Medium, and High — to show how sensitive pricing outcomes are to rider behaviour.

**Actionable Recommendation**

Across thousands of simulations, $0.16 per minute emerges as the most robust pilot price when balancing revenue growth and behavioural risk.

## Methodology

Posterior Predictive Checks (PPCs) to test model fit and catch mis‑specification

Hierarchical priors to handle missing user‑level attributes and reduce overfitting

Monte Carlo simulation using industry elasticity benchmarks and uncertainty‑aware revenue projections

Adversarial model critique to stress‑test assumptions and keep the model honest

## Why This Project Matters

Cyclistic needs prices that are fair, sustainable, and grounded in real behaviour. Mobility data is messy: trip durations are long‑tailed, riders behave differently, and averages hide important uncertainty.

This project matters because it:

- replaces point estimates with full probability distributions

- models revenue in a way that respects how membership fees actually work

- simulates pricing outcomes under behavioural uncertainty

- supports decisions with risk‑adjusted insights, not single‑number forecasts

- demonstrates a modern Bayesian workflow with PPCs, hierarchical priors, and adversarial critique

## Key Takeaways

- Bayesian modelling gives a richer picture than classical averages, especially with skewed data

 Revenue comparisons between rider types need proper accounting, not raw totals

 Pricing decisions benefit from uncertainty‑aware simulation

 The recommended pilot price balances revenue growth with behavioural risk

Future Work

- Seasonal or weather‑based hierarchical models to capture seasonal shifts in behaviour

- Mixture models to identify behavioural subgroups within Casual riders

- Elasticity estimation from historical experiments to replace industry benchmarks

- Spatial modelling to capture station‑level differences

- Full Bayesian decision analysis for expected‑utility optimisation

- Bayesian bandits for dynamic pricing using Thompson Sampling

- Interactive dashboard (Streamlit or Shiny) for exploring posterior distributions and pricing scenarios
