This project applies Bayesian modelling to understand Cyclistic rider behaviour and evaluate pricing strategy under uncertainty. Using PyMC and ArviZ, the analysis models trip duration, value‑per‑trip, and revenue outcomes with full posterior distributions rather than point estimates. A Monte Carlo pricing simulation explores how different elasticity assumptions affect future revenue, leading to a risk‑adjusted recommendation for a $0.16 per‑minute pilot price. The result is a transparent, uncertainty‑aware framework for data‑driven pricing decisions.

## Project Highlights

**Bayesian Inference for Rider Behaviour**  
Trip duration and revenue are modelled using Log‑Normal likelihoods to handle right‑skewed behaviour typical of mobility data.

**Value‑Per‑Trip Economics**  
Resolves the accounting challenge of comparing Casual vs Member revenue by simulating amortised membership fees across realistic usage patterns.

**Monte Carlo Pricing Simulation**  
Forecasts future revenue under multiple price‑elasticity assumptions (Low / Medium / High), enabling scenario‑based decision‑making.

**Actionable Pricing Recommendation**  
A risk‑adjusted pilot price of $0.16 per minute emerges as the most robust strategy across posterior simulations.

## Methodology

- Posterior Predictive Checks (PPCs) to validate model fit and detect mis‑specification.

- Hierarchical priors to accommodate missing user‑level attributes and reduce overfitting.

- Monte Carlo simulation of price elasticity using industry benchmarks and uncertainty‑aware revenue projections.

- Adversarial model critique to ensure assumptions are transparent, defensible, and aligned with domain constraints.

## Why This Project Matters

Cyclistic’s business model depends on balancing accessibility with sustainable revenue. Traditional analytics often rely on averages that obscure uncertainty, skewness, and behavioural variability — especially in mobility data where trip durations are long‑tailed and user types behave differently.

This project matters because the project:

- Replaces point estimates with full probability distributions, giving a more honest view of uncertainty.

- Models revenue in a way that respects accounting reality, especially the amortisation of membership fees.

- Simulates pricing outcomes under behavioural uncertainty, rather than assuming fixed elasticity.

- Supports decision‑makers with risk‑adjusted insights, not just single-number forecasts.

- Demonstrates modern Bayesian workflow skills, including PPCs, hierarchical priors, and adversarial critique.

For recruiters and stakeholders, this project showcases analytical maturity, statistical honesty, and the ability to translate modelling into actionable business recommendations.

## Key Takeaways

- Bayesian modelling provides richer insight than classical averages, especially with skewed behavioural data.

- Revenue comparisons between rider types require careful accounting treatment — not raw trip totals.

- Pricing decisions benefit from uncertainty‑aware simulation rather than single‑point forecasts.

The recommended pricing pilot balances revenue growth with behavioural risk.

## Future Work

- Hierarchical modelling by season or weather
  Capture seasonal variation in rider behaviour.

- Mixture models for trip duration  
  Identify potential behavioural subgroups within Casual riders.

- Elasticity estimation from historical experiments  
  Replace industry benchmarks with empirical elasticity from A/B tests.

- Integration with spatial data  
  Model station‑level differences and neighbourhood‑specific sensitivity.

- Full Bayesian decision analysis  
  Move from scenario simulation to formal expected‑utility optimisation.

- Bayesian bandit algorithms for dynamic pricing  
  Use multi‑armed bandit methods (e.g., Thompson Sampling) to dynamically optimise pricing in live experiments by       balancing exploration and exploitation.

- Interactive dashboard  
  Build a Streamlit or Shiny app for exploring posterior distributions and pricing scenarios.
