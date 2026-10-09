# Frequency Containment Reserves (FCR) and Droop Control

Power system stability requires continuous equilibrium between active power generation and consumption. Any frequency deviation from the nominal setpoint (50 Hz in Europe, 60 Hz in North America) signals a power imbalance. Frequency Containment Reserves (FCR), historically termed primary frequency control, constitute the first line of defense against frequency excursions.

## Droop Control and Governor Response

Traditional synchronous generators provide primary frequency response via mechanical turbine governors tuned with a proportional droop characteristic. Droop percentage (s) defines the ratio of relative frequency deviation to relative power change:
s = (delta_f / f_nominal) / (delta_P / P_rated).

A typical 4% or 5% droop setting dictates that a 1% drop in grid frequency causes a 20% to 25% increase in generator active power output. Modern inverter-based resources (IBRs) emulate this governor behavior through software-configured virtual droop control loops in their digital signal processors (DSPs), reacting within milliseconds rather than seconds.

## Operational Dynamics of FCR-N and FCR-D

In many grid codes, FCR is partitioned into normal operation reserves (FCR-N) and disturbance reserves (FCR-D):
- FCR-N operates within a narrow frequency band (e.g., 49.90 Hz to 50.10 Hz in the Nordic synchronous area), continuously stabilizing minute-by-minute system load noise.
- FCR-D activates only when major disturbances drive frequency beyond normal thresholds (e.g., dropping below 49.90 Hz due to a tripped nuclear unit or transmission tie-line trip).

FCR delivery requires full power activation within 15 to 30 seconds and sustained delivery for at least 15 minutes. Stationary battery systems providing FCR must manage their state of charge continuously, utilizing energy trading in intraday markets to avoid saturation or depletion during prolonged frequency shifts.
## Reserve Market Dimensioning and Cross-Border Sharing

Transmission system operators dimension FCR reserve pools based on the reference incident: the sudden loss of the single largest generation unit, high-voltage direct current (HVDC) interconnector, or industrial load block in the synchronous area (typically 3,000 MW in Continental Europe). TSOs procure symmetric bidirectional FCR capacity through competitive daily auctions. Under the European FCR Cooperation framework, neighboring TSOs share reserve obligations across bidding zones using available cross-border transmission margins, minimizing overall procurement costs while ensuring sufficient local reserve distribution to prevent inter-area power oscillations.
