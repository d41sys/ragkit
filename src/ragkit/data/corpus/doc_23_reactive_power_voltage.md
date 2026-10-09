# Reactive Power Compensation and Volt-VAR Optimization

In alternating current (AC) power systems, active power (watts) performs useful mechanical or thermal work, while reactive power (volt-amperes reactive, VAR) supports the magnetic and electric fields necessary to maintain voltage profiles across inductive transmission and distribution lines.

## The Physics of Voltage Control and Line Reactance

Voltage drops along an AC transmission line are governed primarily by the flow of reactive power across inductive line reactance (X):
delta_V ~= (R * P + X * Q) / V.

Because high-voltage transmission lines exhibit high X/R ratios, injecting inductive reactive power lowers local voltage, whereas injecting capacitive reactive power raises voltage. Maintaining voltages within strict operational bounds (typically +/- 5% of nominal) prevents insulation breakdown and motor stall phenomena (fault-induced delayed voltage recovery / FIDVR).

## Volt-VAR Optimization (VVO) and Smart Inverter Control

Distribution utilities implement Volt-VAR Optimization (VVO) by coordinating legacy hardware—such as on-load tap changers (OLTC) at substations, switched capacitor banks, and step voltage regulators—with distributed solar and battery smart inverters. Modern grid codes (IEEE 1547-2018) mandate that distributed inverters support automated Volt-VAR curves:
- When local terminal voltage is low, the inverter injects capacitive reactive power.
- When voltage rises due to heavy solar generation, the inverter absorbs reactive power.

VVO software algorithms continuously solve non-linear AC optimal power flow equations, coordinating thousands of inverters to minimize system losses, smooth voltage fluctuations, and execute conservation voltage reduction (CVR) to lower peak feeder demand.
## High-Voltage Transmission Shunt Compensation: STATCOM and SVC

On bulk extra-high-voltage (EHV) transmission grids, reactive power compensation is provided by dynamic power electronic shunt devices: Static Synchronous Compensators (STATCOM) and Static VAR Compensators (SVC). Unlike switched capacitor banks whose reactive output drops quadratically with system voltage (Q = V^2 / X_C), STATCOMs utilize voltage-source converters capable of maintaining constant rated capacitive current injection even during severe voltage depressions down to 0.2 per unit. This continuous dynamic voltage support prevents voltage collapse during post-contingency transmission faults, allowing operators to push higher power transfer flows across long-distance transmission corridors.
