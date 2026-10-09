# Battery Energy Storage System (BESS) Dispatch and Degradation Management

Grid-scale Battery Energy Storage Systems (BESS), predominantly utilizing lithium iron phosphate (LFP) and nickel manganese cobalt (NMC) chemistries, provide multi-service revenue stacking across energy arbitrage, primary frequency regulation, and capacity markets. Effective dispatch optimization requires co-optimizing market revenues against electrochemical degradation costs.

## State of Charge (SoC) Dynamics and Round-Trip Efficiency

The operational envelope of a stationary BESS is bounded by physical state of charge limits, C-rate capabilities, and auxiliary parasitic loads (e.g., HVAC thermal management and inverter cooling). Round-trip efficiency (RTE) typically spans 85% to 92% at the AC interconnection point. Dispatch algorithms formulated via mixed-integer linear programming (MILP) enforce dynamic SoC state transitions:
SoC(t) = SoC(t-1) + [P_charge(t) * eta_charge - P_discharge(t) / eta_discharge] * delta_t / E_nominal.

Simultaneous charging and discharging is strictly prohibited using binary operational variables. Furthermore, depth of discharge (DoD) restrictions prevent excursion beyond 10% to 90% SoC to preserve cell longevity.

## Degradation Modeling and Multi-Service Stacking

Battery degradation manifests as both calendar aging (time, ambient temperature, resting SoC) and cyclic aging (throughput, C-rate, DoD per cycle). Modern battery dispatch platforms incorporate semi-empirical degradation cost curves into the objective function. The marginal cost of cell wear is priced as a degradation penalty per megawatt-hour throughput:
C_deg = [C_replacement / (Cycle_life(DoD) * E_nominal)] * P_discharge * delta_t.

When participating in frequency containment reserves (FCR), state-of-charge restoration algorithms utilize deadband zones and continuous micro-trades in intraday markets to keep resting SoC centered near 50%, minimizing resting degradation while preserving bidirectional reserve availability.
## Thermal Management, Auxiliary Power, and Safety Systems

The operational life and round-trip efficiency of grid-scale BESS are governed by environmental thermal dynamics. Liquid glycol cooling plates and HVAC heat pumps maintain internal cell temperatures within optimal 20°C to 25°C operating bands. Auxiliary parasitic loads from thermal management systems consume between 2% and 6% of gross system throughput depending on ambient climate conditions. Thermal runaway mitigation mandates multi-tier safety architectures compliant with NFPA 855 and UL 9540A testing standards, integrating off-gas sniffing sensors, aerosol fire suppression, and deflagration venting panels. Dispatch controllers monitor continuous internal cell resistance changes and electrochemical impedance spectroscopy (EIS) signatures to detect premature lithium plating during cold-temperature high-C-rate charging events.
