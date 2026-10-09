# Locational Marginal Pricing (LMP) and Nodal Congestion

In nodal electricity markets (such as PJM, ERCOT, CAISO, and ISO-NE), wholesale prices are calculated at individual transmission substations through Locational Marginal Pricing (LMP). LMP calculates the cost to supply the next megawatt of demand at a specific physical bus, reflecting transmission losses and physical transmission congestion.

## The Three Components of LMP

The LMP at bus i is mathematically decomposed into three distinct additive components:
LMP_i = Energy_Component + Congestion_Component_i + Loss_Component_i.

1. System Energy Price: The uniform marginal cost of generating the next MWh of energy from the unconstrained system reference bus.
2. Marginal Congestion Component (MCC): The economic cost of binding transmission constraints. When a transmission line reaches its thermal or stability limit, the system operator must dispatch out-of-merit generators downstream to relieve the line, while backing down cheaper generators upstream. The MCC becomes negative upstream of the constraint (low prices or negative prices due to local surplus) and positive downstream (high prices reflecting local scarcity).
3. Marginal Loss Component (MLC): The marginal cost of transmission resistance losses incurred by moving power from the reference bus to bus i.

## Financial Transmission Rights (FTRs) and Flexibility Siting

LMP price divergence creates financial risk for market participants. Generators and loads hedge spatial price differentials using Financial Transmission Rights (FTRs) or Congestion Revenue Rights (CRRs). Distributed energy resources and battery storage developers utilize historical LMP nodal data to identify chronic congestion nodes, siting flexible assets where the spread between local off-peak and on-peak prices is maximized.
## Shift Factors and Transmission Sensitivity Matrices

The calculation of Locational Marginal Prices in security-constrained economic dispatch (SCED) relies on linear sensitivity matrices known as Generation Shift Factors (GSFs) or Power Transfer Distribution Factors (PTDFs). The PTDF matrix quantifies the fraction of power injected at bus i that flows across a specific transmission line l:
Line_Flow_l = sum_i [ PTDF(l, i) * Net_Injection_i ].

When a transmission line reaches its thermal rating, the dual shadow price (Lagrange multiplier) of that line constraint becomes non-zero. The marginal congestion component at any bus is computed as the product of the shadow price and the bus's shift factor relative to the reference bus, providing an exact mathematical signal of each generator's marginal impact on physical grid congestion.
