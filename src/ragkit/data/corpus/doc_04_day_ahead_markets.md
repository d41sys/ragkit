# Day-Ahead Electricity Market Clearing and Merit Order Dispatch

Wholesale electricity markets rely on the day-ahead market (DAM) as the primary clearinghouse for scheduled power delivery. Operating 12 to 36 hours prior to the physical delivery hour, the day-ahead auction clears hourly or half-hourly energy contracts through uniform marginal pricing algorithms.

## The Merit Order Effect and Marginal Cost Pricing

The fundamental economic mechanism of the day-ahead market is the merit order dispatch curve. Generation assets and demand bids are ranked in ascending order of short-run marginal cost (SRMC). 

Variable renewable generation, such as wind and solar, possesses near-zero marginal fuel costs and therefore occupies the far left of the merit order supply stack. As renewable output expands, it displaces higher-marginal-cost thermal assets—such as combined-cycle gas turbines (CCGT) and open-cycle peaker plants—shifting the market clearing intersection to the right and downward. This phenomenon is termed the merit order effect. The market clearing price (system marginal price) is established by the bid of the last and most expensive generation unit required to satisfy aggregated demand. All dispatched inframarginal generators receive this single uniform clearing price.

## Market Coupling and Algorithmic Clearing (EUPHEMIA)

In integrated power systems such as the European Single Day-Ahead Coupling (SDAC), market clearing is executed via complex mathematical optimization engines like EUPHEMIA (Pan-European Hybrid Electricity Market Integration Algorithm). EUPHEMIA resolves welfare maximization across dozens of bidding zones subject to cross-border transmission capacities (Net Transfer Capacity or Flow-Based Market Coupling). When transmission interconnectors become congested, price divergence occurs between bidding zones, signaling the locational scarcity of power. Market participants submit complex orders, including block bids, linked orders, and minimum income conditions (MIC), necessitating mixed-integer quadratic programming to ensure economic feasibility and prevent dispatch paradoxes.
## Financial Settlement, Block Orders, and Dual Cleared Auctions

Day-ahead market clearing balances physical security constraints with complex participant bidding formats. Market participants submit both simple hourly price-volume pairs and sophisticated multi-hour block bids that specify minimum execution durations and fill-or-kill conditions. EUPHEMIA enforces strict welfare maximization principles while penalizing paradoxically accepted bids. Following market clearing, market operators calculate locational financial imbalances and publish binding hourly generation schedules to balancing responsible parties, serving as the commercial baseline for real-time dispatch and transmission congestion management across European and North American wholesale interconnections.
