# Distribution Grid Congestion Management and Local Flexibility Markets

The proliferation of rooftop photovoltaics, electric vehicle fast chargers, and heat pumps creates operational challenges on low- and medium-voltage distribution networks. Distribution System Operators (DSOs) traditionally resolved thermal overloading and voltage violations through capital-intensive copper-and-iron grid reinforcement. Local flexibility markets provide an operational alternative.

## Distribution Grid Bottlenecks: Thermal and Voltage Limits

Unlike transmission grids, distribution networks have low X/R ratios (higher resistance relative to reactance), meaning that active power injections directly impact nodal voltage magnitudes:
delta_V ~= (R * P + X * Q) / V_nominal.

During sunny midday hours, reverse power flows from distributed PV can drive voltage above statutory limits (+10% over nominal). Conversely, evening EV charging clusters cause thermal overloads on local distribution transformers and feeder cables.

## Local Flexibility Market Mechanisms

Local flexibility platforms (e.g., Piclo Flex, GOPACS, NODES) connect DSOs with local flexibility providers and aggregators. The DSO identifies a localized network constraint:
1. Day-ahead or intraday capacity procurement: The DSO broadcasts a flexibility request for a specific feeder node, specifying required MW curtailment or injection over a designated time window.
2. Market Clearing: Aggregators submit price-volume bids from local assets (residential BESS, commercial HVAC, EV chargers). The platform clears bids that provide the most cost-effective relief.
3. Baseline and Settlement: After delivery, meter telemetry is verified against baseline profiles, and providers receive availability and activation compensation, avoiding or deferring expensive asset replacement by the DSO.
## Dynamic Voltage Regulation and Reactive Power Droop (Volt-Watt and Volt-VAR)

To manage high concentrations of distributed photovoltaic generation, distribution network operators configure automated autonomous inverter response curves directly in customer firmware. Under Volt-VAR droop control (Q(V)), distributed inverters absorb inductive reactive power when local terminal voltage exceeds statutory limits, pulling down voltage without sacrificing active power revenue. When severe voltage elevation persists, autonomous Volt-Watt curves (P(V)) engage, linearly throttling active power output as voltage rises toward inverter disconnect trip limits. By coordinating autonomous inverter curves with local flexibility tenders, DSOs maintain power quality and minimize renewable curtailment on low-voltage suburban feeders.
