# Imbalance Settlement Principles and Dual-Pricing Mechanisms

Imbalance settlement is the post-delivery commercial accounting framework that financially penalizes deviations between a Balance Responsible Party's (BRP) contracted market position and its actual metered energy flows. Balancing market integrity relies on this mechanism to incentivize market participants to maintain self-balancing.

## Balance Responsible Parties and Metered Deviations

Every grid interconnection point is assigned to a designated Balance Responsible Party (BRP). A BRP's imbalance volume in settlement period t is defined as:
Imbalance_Volume(t) = Metered_Generation(t) - Metered_Consumption(t) + Contracted_Sales(t) - Contracted_Purchases(t).

An imbalance is positive (long position) when the BRP injects more energy into the system than its net contractual obligation; conversely, an imbalance is negative (short position) when consumption exceeds scheduled purchases or generation falls short.

## Single vs. Dual Pricing Regimes

Imbalance pricing frameworks determine the financial penalty applied per MWh of deviation:

1. Single Pricing Regime: All imbalances in a given settlement period are settled at the same price, regardless of the BRP's individual direction. The price reflects the weighted average or marginal cost of balancing energy activated by the Transmission System Operator (TSO). In single pricing, BRPs are incentivized to support the system: if the overall grid is short, being long is rewarded at the elevated balancing energy price.

2. Dual Pricing Regime: Distinct prices apply depending on whether the individual BRP's deviation aligns with or opposes the aggregate system imbalance direction. If a BRP is short while the entire grid is short, it pays the high marginal activated balancing energy price. However, if the BRP is long during a system-short event, it may only receive the day-ahead spot price. Dual pricing prevents gaming and speculative leaning on the TSO's reserves, enforcing rigorous adherence to scheduled nominations across each 15-minute imbalance settlement period (ISP).
## Financial Settlement Timelines and Accounting Harmonization

Imbalance accounting operates on multi-stage financial settlement timelines. Initial provisional settlement occurs within 5 to 10 business days following the delivery month, utilizing validated raw meter data and activated balancing energy logs. Final financial reconciliation takes place after 60 to 90 days following residual profiling corrections and historical metering reconciliations. Harmonized European balancing frameworks (MARI and PICASSO platforms) standardize activated balancing reserves across borders, establishing uniform merit order clearing for Frequency Restoration Reserves (aFRR and mFRR) and decoupling national settlement from physical cross-border transmission flows.
