# Peak Shaving and Transmission Network Charge Optimization

For large commercial and industrial (C&I) facilities, electricity bills are heavily driven by capacity and demand charges rather than volumetric energy consumption alone. Peak shaving strategies utilize behind-the-meter assets to suppress maximum demand peaks, drastically reducing network tariff obligations.

## Tariff Architecture: Capacity Charges and Coincident Peaks

Network distribution utilities and transmission system operators bill commercial customers using two distinct peak charge models:
1. Maximum Non-Coincident Peak Demand: Charges assessed on the customer's highest single 15-minute average power draw (kW or kVA) recorded across the monthly billing cycle, regardless of what the overall grid was doing.
2. Coincident Peak Charges: Charges assessed on customer consumption during specific system-wide peak hours. In the PJM interconnection (5CP) and ERCOT (4CP), transmission charges for the subsequent year are determined by a facility's average consumption during the four or five highest grid peak hours during the summer.

## Optimization Strategies and Behind-the-Meter BESS Dispatch

Industrial facilities deploy BESS and load shedding automation to execute peak shaving. The control algorithm continuously monitors the facility's main utility meter. When real-time consumption approaches a predetermined threshold setpoint, the battery instantaneously discharges, supplying load from the battery rather than drawing from the substation:
P_grid(t) = P_load(t) - P_battery_discharge(t).

Setting the threshold setpoint involves a stochastic optimization trade-off. If the setpoint is set too aggressively low, the battery may deplete its available capacity early in a prolonged demand peak, leaving the facility vulnerable to a high peak draw in the subsequent hour. Sophisticated energy management systems incorporate day-ahead production schedules and load forecasting to dynamically tune the daily shaving threshold.
## Power Factor Correction and Apparent Power Demand Charges

Industrial peak demand billing frequently incorporates penalties for low power factor, assessing charges on peak apparent power (kVA) rather than active power (kW) alone. When industrial induction motors or electric arc furnaces draw heavy inductive loads, the facility's power factor declines below utility thresholds (typically 0.90 to 0.95 lag). Advanced peak shaving systems co-optimize battery inverters to deliver four-quadrant active and reactive power support. By synthesizing capacitive reactive power (VAR injection) simultaneously with active power discharge, the battery system raises the facility's operating power factor, eliminating reactive demand surcharges while flattening peak kilowatt demand profiles.
