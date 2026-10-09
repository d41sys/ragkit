# Industrial and Commercial Demand Response Mechanisms

Demand response (DR) is the intentional alteration of electric consumption patterns by end-use customers in response to economic price signals or grid reliability mandates. In electricity systems facing high penetration of variable renewable energy (VRE), demand response serves as a non-generation balancing asset capable of absorbing excess generation or shaving critical system peak loads.

## Operational Categories and Signal Types

Demand response programs are bifurcated into two primary paradigms: price-based and incentive-based programs.

Price-based demand response includes time-of-use (TOU) tariffs, critical peak pricing (CPP), and real-time pricing (RTP). Under RTP, industrial consumers with high thermal mass, flexible batch production schedules, or chlor-alkali electrolysis processes adjust hourly consumption based on day-ahead or intraday spot prices. When power prices spike due to transmission constraints or generator outages, automated building management systems (BMS) and programmable logic controllers (PLCs) ramp down non-essential chillers, fans, and pump loads.

Incentive-based demand response involves formal capacity and energy contracts. Participants contractually agree to curtail predefined megawatt (MW) blocks within notification windows ranging from 10 minutes (fast contingency DR) to several hours (economic capacity DR). In return, participants receive fixed availability payments (capacity reservations) plus energy activation payments calculated relative to a customer baseline load (CBL).

## Verification and Customer Baseline Methodology

A central challenge in demand response administration is establishing the counterfactual consumption baseline. System operators commonly calculate the customer baseline using the High 4-of-5 or 10-of-10 rolling non-event day methodology. An asymmetric morning adjustment factor is frequently applied to capture weather-dependent deviations. If a facility delivers load reductions during an event, the discrepancy between actual meter data and the CBL determines financial settlement. Sub-metering verification standards ensure that behind-the-meter diesel backup generation is distinguished from authentic physical load curtailment, enforcing environmental emissions compliance.
## Grid Codes and Telemetry Architecture

Participation in industrial demand response requires high-resolution telemetry architectures conforming to strict system operator standards. Remote terminal units (RTUs) and smart power quality meters record active and reactive power at 1-second to 4-second intervals, streaming encrypted operational telemetry via IEC 60870-5-104 or DNP3 protocols directly to the transmission or distribution control center. Facilities must demonstrate automated event execution without manual operator latency, utilizing secure supervisory control and data acquisition (SCADA) handshakes. Baseline models must account for seasonal manufacturing variability, scheduled maintenance downtime, and ambient wet-bulb temperature variations to ensure audited load reductions reflect authentic demand adjustments rather than routine operational idling.
